"""Forecaster: shared encoder, channel IDs, masking, fusion with cross-attention to the throttle, one forecast per channel."""

from collections.abc import Callable

import numpy as np
import torch
from torch import nn

from contracts import CHANNELS, D_EMBED, WINDOW, Run, Verdict
from model.data import HORIZON, Normalizer, make_windows
from model.encoder import ChannelEncoder

# Attention heads in the fusion block
N_HEADS = 4
# Feed-forward width: 16 -> 64 -> 16
FFN_HIDDEN = 64
# Transformer blocks, 1 to 2. Start with 1
N_BLOCKS = 1
# Windows per scoring batch. Bounds memory, especially when masking makes 8 copies of each
SCORE_BATCH = 2048

# Throttle samples per command token. With horizon 10, the last token is the forecast gap
THROTTLE_CHUNK = 10

# How per-channel residuals combine into one score. Compared against "max" on val at Gate 1
AGGREGATE = "mean"


class Forecaster(nn.Module):
    """Forecasts every channel `horizon` samples ahead from all channels' windows and the throttle history, with optional masking."""

    def __init__(self, horizon: int = HORIZON, command_input: bool = True) -> None:
        super().__init__()
        # The throttle history must split into whole chunks, or the last token would straddle the target
        if (WINDOW + horizon) % THROTTLE_CHUNK:
            raise ValueError(f"WINDOW + horizon must be a multiple of {THROTTLE_CHUNK}, not {WINDOW + horizon}")
        self.horizon = horizon
        # False is the D12 ablation: every throttle history becomes zeros, so cross-attention runs but carries no
        # command information. Not a weight, so the same seed builds the same weights either way
        self.command_input = command_input
        # Command tokens per throttle history. 11 with horizon 10
        self.n_command_tokens = (WINDOW + horizon) // THROTTLE_CHUNK
        self.encoder = ChannelEncoder()
        self.channel_id = nn.Parameter(torch.randn(len(CHANNELS), D_EMBED) * 0.1)
        self.missing = nn.Parameter(torch.zeros(D_EMBED))
        # Each 10-sample throttle chunk becomes one 16-number command token
        self.throttle_embed = nn.Linear(THROTTLE_CHUNK, D_EMBED)
        # Where each chunk sits in time. Built from the horizon, so a model rebuilt by load_run matches its weights
        self.throttle_pos = nn.Parameter(torch.randn(self.n_command_tokens, D_EMBED) * 0.1)
        # Self-attention across channels, then cross-attention to the command tokens, then feed-forward
        layer = nn.TransformerDecoderLayer(D_EMBED, N_HEADS, FFN_HIDDEN, dropout=0.0, batch_first=True)
        self.fusion = nn.TransformerDecoder(layer, N_BLOCKS)
        # Each channel's fused token -> its own forecast
        self.head = nn.Linear(D_EMBED, 1)

    def tokens(self, windows: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encoder output, masked channels replaced by `missing`, plus channel IDs. (batch, 8, 16)."""
        # (batch, 8, 16)
        t = self.encoder(windows)

        if mask is not None:
            # hide masked channels
            t = torch.where(mask.unsqueeze(-1), self.missing, t)

        # tag every token with its channel
        return t + self.channel_id

    def command_tokens(self, throttle_hist: torch.Tensor) -> torch.Tensor:
        """(batch, WINDOW + horizon) throttle history to (batch, chunks, 16) command tokens."""
        expected = self.n_command_tokens * THROTTLE_CHUNK
        # A history cut for another horizon would be silently misaligned
        if throttle_hist.shape[-1] != expected:
            raise ValueError(f"throttle_hist must have {expected} samples, not {throttle_hist.shape[-1]}")
        # (batch, chunks, 10): consecutive samples, oldest chunk first, the forecast gap last.
        # The chunk count is explicit, so an empty batch still reshapes cleanly
        chunks = throttle_hist.reshape(len(throttle_hist), self.n_command_tokens, THROTTLE_CHUNK)
        return self.throttle_embed(chunks) + self.throttle_pos

    def forward(self, windows: torch.Tensor, throttle_hist: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """(batch, 8, WINDOW) plus (batch, WINDOW + horizon) throttle to (batch, 8): each channel `horizon` samples ahead."""
        # The one place the ablation differs. Training, scoring and attention all come through here
        if not self.command_input:
            throttle_hist = torch.zeros_like(throttle_hist)
        # Channel tokens are the queries (tgt), command tokens the keys and values (memory)
        fused = self.fusion(self.tokens(windows, mask), self.command_tokens(throttle_hist))

        # (batch, 8): one forecast per channel
        return self.head(fused).squeeze(-1)


def floor_weighted_mse(pred: torch.Tensor, target: torch.Tensor, floor: torch.Tensor) -> torch.Tensor:
    """Each channel's squared error in multiples of its noise floor, averaged."""
    return ((pred - target) ** 2 / floor).mean()


@torch.no_grad()
def forecast_run(model: Forecaster, normalizer: Normalizer, run: Run, masked: bool = True) -> np.ndarray:
    """Forecasts for a whole run, (N_STEPS, 8), normalized, aligned to target times. NaN where none exists."""
    x = normalizer.transform(run.sensors)
    windows, throttle_hist, _ = make_windows(x, run.throttle, model.horizon)
    n_ch = len(CHANNELS)

    preds = []
    for lo in range(0, len(windows), SCORE_BATCH):
        w = torch.from_numpy(windows[lo:lo + SCORE_BATCH].astype(np.float32))
        th = torch.from_numpy(throttle_hist[lo:lo + SCORE_BATCH].astype(np.float32))
        if masked:
            b = len(w)
            # Each window 8 times in a row
            w8 = w.repeat_interleave(n_ch, dim=0)
            th8 = th.repeat_interleave(n_ch, dim=0)
            # Copy c hides channel c
            mask = torch.eye(n_ch, dtype=torch.bool).repeat(b, 1)
            # Indexed [window, copy, channel]
            p = model(w8, th8, mask=mask).reshape(b, n_ch, n_ch)
            # Copy c's forecast of channel c
            p = p.diagonal(dim1=1, dim2=2)
        else:
            p = model(w, th)
        preds.append(p.numpy())

    out = np.full((len(x), n_ch), np.nan)
    out[WINDOW - 1 + model.horizon:] = np.concatenate(preds)
    return out


@torch.no_grad()
def attention_by_channel(model: Forecaster, normalizer: Normalizer, run: Run) -> np.ndarray:
    """(N_STEPS, 8): the self-attention each channel receives, from one unmasked pass. NaN where no forecast exists.

    Averaged over heads, querying channels and blocks. Unmasked, since in a masked pass the scored channel's own
    token is hidden. Decision log, 2026-11-20 (attribution).
    """
    x = normalizer.transform(run.sensors)
    windows, throttle_hist, _ = make_windows(x, run.throttle, model.horizon)
    # Self-attention only. Cross-attention goes to the command tokens, so it can't rank channels
    layers = [layer.self_attn for layer in model.fusion.layers]
    captured: list[list[torch.Tensor]] = [[] for _ in layers]

    # The decoder layer asks for no weights, so this switches them on for the call, averaged over heads
    def ask_for_weights(module, args, kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = True
        return args, kwargs

    handles = []
    for i, attn in enumerate(layers):
        handles.append(attn.register_forward_pre_hook(ask_for_weights, with_kwargs=True))
        # output is (attended values, weights), the weights shaped (batch, 8 queries, 8 keys)
        handles.append(attn.register_forward_hook(lambda module, args, output, i=i: captured[i].append(output[1])))
    try:
        for lo in range(0, len(windows), SCORE_BATCH):
            w = torch.from_numpy(windows[lo:lo + SCORE_BATCH].astype(np.float32))
            th = torch.from_numpy(throttle_hist[lo:lo + SCORE_BATCH].astype(np.float32))
            model(w, th)
    finally:
        # Always removed, so no later forecast carries the hooks
        for h in handles:
            h.remove()

    # (n, queries, keys) averaged over queries gives what each key channel receives, then averaged over blocks
    received = np.mean([torch.cat(c).mean(dim=1).numpy() for c in captured], axis=0)
    out = np.full((len(x), len(CHANNELS)), np.nan)
    out[WINDOW - 1 + model.horizon:] = received
    return out


def residual_by_channel(model: Forecaster, normalizer: Normalizer, run: Run, masked: bool = True) -> np.ndarray:
    """(N_STEPS, 8): each channel's squared miss in noise-floor units. NaN where there's no forecast."""
    x = normalizer.transform(run.sensors)
    forecast = forecast_run(model, normalizer, run, masked=masked)
    return (x - forecast) ** 2 / normalizer.floor


def aggregate(r: np.ndarray, how: str = AGGREGATE) -> np.ndarray:
    """(N_STEPS, 8) floor-weighted residuals to (N_STEPS,) scores. 0 where there's no forecast."""
    if how == "mean":
        combined = r.mean(axis=1)
    elif how == "max":
        combined = r.max(axis=1)
    else:
        raise ValueError(f"unknown aggregate {how!r}")
    return np.nan_to_num(combined, nan=0.0)


def residual_scores(
    model: Forecaster, normalizer: Normalizer, run: Run, masked: bool = True, how: str = AGGREGATE
) -> np.ndarray:
    """Per-sample anomaly score, (N_STEPS,). 0 before the first forecast."""
    return aggregate(residual_by_channel(model, normalizer, run, masked), how)


def to_verdict(
    model: Forecaster,
    normalizer: Normalizer,
    run: Run,
    threshold: float,
    confidence: Callable[[float], float],
    masked: bool = True,
    how: str = AGGREGATE,
    score: Callable[[np.ndarray, Run], np.ndarray] | None = None,
) -> Verdict:
    """Flag, confidence and channels ranked by residual.

    score turns the per-channel residuals and the run into one score per sample. The v1 scoring is the std score,
    passed in as StandardizedScore.verdict_score (decision log, 2026-11-23). Without it, the verdict uses the
    aggregate named by how, as on D6 to D9.

    threshold and confidence must both come from the same detector's calibrate peaks, so a run flags exactly
    when its confidence is at least 1 minus the budget. Mixing detectors would show "alarm, confidence 0.97".
    confidence is the share of healthy calibrate runs this run is more unusual than, not the probability of a fault.
    It's passed in as a function, so model/ needs nothing from evaluation/.
    """
    # Same switches as residual_scores, so the harness and the verdict always score a run the same way
    r = residual_by_channel(model, normalizer, run, masked)
    scores = aggregate(r, how) if score is None else score(r, run)
    # The std score's blind rows are -inf, so they can never be the peak
    peak = scores.max()
    # A NumPy bool would break JSON and `is True`, so convert at the contract boundary
    flag = bool(peak > threshold)
    # Each channel's average miss, skipping the rows with no forecast
    per_channel = np.nanmean(r, axis=0)
    ranked = tuple(CHANNELS[i] for i in np.argsort(per_channel)[::-1])
    # A plain float at the contract boundary, for the same reason as the flag
    return Verdict(flag=flag, confidence=float(confidence(peak)), ranked_channels=ranked)
