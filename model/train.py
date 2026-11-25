"""Train the forecaster on healthy train runs, with random channel masking. Saves and loads runs/<name>/.

python -m model.train                  trains the default model into runs/h10_mask0.25_cmd_seed0/
python -m model.train --mask-prob 0    trains the no-masking comparison model into runs/h10_mask0_cmd_seed0/
python -m model.train --no-command     trains the D12 ablation into runs/h10_mask0.25_nocmd_seed0/
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from contracts import CHANNELS
from generator.splits import SPLITS, make_run
from model.data import HORIZON, Normalizer, make_windows
from model.encoder import ENCODER_HIDDEN
from model.forecaster import (
    AGGREGATE,
    FFN_HIDDEN,
    N_BLOCKS,
    N_HEADS,
    THROTTLE_CHUNK,
    Forecaster,
    floor_weighted_mse,
)

# Train runs used. Start small and scale to 2,000 if the model underfits
N_TRAIN_RUNS = 500
# Windows per training step
BATCH = 256
# Optimizer steps
STEPS = 10_000
# Adam learning rate
LR = 1e-3
# Chance each channel is hidden in a training window
MASK_PROB = 0.25
# Steps between probe measurements
LOG_EVERY = 100
# Windows in the fixed probe batch. Bigger than a training batch, so the curve is smooth
PROBE_BATCH = 1024
# Where trained models go. Git keeps each run's config and metrics, not its weights
RUNS_DIR = Path("runs")


@dataclass
class TrainResult:
    """The trained model, its normalizer, the probe loss curve, and every setting train() used."""

    model: Forecaster
    normalizer: Normalizer
    # Loss on one fixed batch. losses[0] is before any training, then one every LOG_EVERY steps
    losses: list[float]
    # Every setting train() actually used, so a saved config can only describe this model
    config: dict[str, Any]
    # Wall-clock training time. 0 when unknown
    seconds: float = 0.0


def random_mask(batch: int, p: float, rng: np.random.Generator) -> torch.Tensor:
    """(batch, 8) bool: each channel hidden with probability p, never all eight."""
    # Written this way so NaN fails too, since every comparison with NaN is False
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], not {p}")
    n_channels = len(CHANNELS)

    # Each channel is hidden on its own with probability p
    mask = rng.random((batch, n_channels)) < p

    # Rows where every channel came out hidden
    all_hidden = np.flatnonzero(mask.all(axis=1))

    # Show one random channel again in each of those rows
    mask[all_hidden, rng.integers(0, n_channels, size=len(all_hidden))] = False
    return torch.from_numpy(mask)


def architecture(command_input: bool = True) -> dict[str, int | bool]:
    """Settings that change how a model runs. A saved model only makes sense with the same ones."""
    return {
        "encoder_hidden": ENCODER_HIDDEN,
        "n_heads": N_HEADS,
        "ffn_hidden": FFN_HIDDEN,
        "n_blocks": N_BLOCKS,
        # D6 configs have neither key, so a model trained before cross-attention is refused with a clear message
        "command_aware": True,
        "throttle_chunk": THROTTLE_CHUNK,
        # False is the D12 ablation, which reads an all-zero throttle history
        "command_input": command_input,
    }


def train(
    horizon: int = HORIZON, mask_prob: float = MASK_PROB, seed: int = 0, steps: int = STEPS, command_input: bool = True
) -> TrainResult:
    """Train on healthy train runs. mask_prob=0 gives the no-masking comparison model. command_input=False the ablation."""
    # Checked here, not only in random_mask, which never runs when mask_prob is 0, negative or NaN
    if not 0.0 <= mask_prob <= 1.0:
        raise ValueError(f"mask_prob must be in [0, 1], not {mask_prob}")
    if steps < 1:
        raise ValueError(f"steps must be at least 1, not {steps}")

    began = time.perf_counter()
    torch.manual_seed(seed)
    # Built first, so every run starts from the same weights whatever else changes
    model = Forecaster(horizon, command_input)

    # Which (run, window) pairs go in each batch. Streams start at 1, since [seed, 0] equals plain seed
    sample_rng = np.random.default_rng([seed, 1])
    # Which channels get hidden. Its own stream, so masking can't change which windows get sampled
    mask_rng = np.random.default_rng([seed, 2])
    # The probe batch. Its own stream, so measuring never shifts training
    probe_rng = np.random.default_rng([seed, 3])

    runs = [make_run("train", s) for s in SPLITS["train"][:N_TRAIN_RUNS]]
    normalizer = Normalizer.fit(runs)
    floor = torch.tensor(normalizer.floor, dtype=torch.float32)

    # Each run normalized once, then cut into window views by the same code scoring uses
    cut = [make_windows(normalizer.transform(r.sensors), r.throttle, horizon) for r in runs]
    # Every run has the same length, so the same number of windows
    n_windows = len(cut[0][0])

    def draw(rng: np.random.Generator, size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Random (run, window) pairs copied out of the views: windows, throttle, targets."""
        run_idx = rng.integers(len(cut), size=size)
        start = rng.integers(n_windows, size=size)

        # Part 0 is windows, 1 is throttle, 2 is targets
        def pick(part: int) -> torch.Tensor:
            return torch.from_numpy(np.stack([cut[r][part][s] for r, s in zip(run_idx, start)]).astype(np.float32))

        return pick(0), pick(1), pick(2)

    def masked(rng: np.random.Generator, size: int) -> torch.Tensor | None:
        # Plain unmasked path when masking is off, so the comparison run is clearly unmasked
        return random_mask(size, mask_prob, rng) if mask_prob > 0 else None

    # One fixed batch and mask, measured before training and every LOG_EVERY steps, so the curve compares like with like
    p_windows, p_throttle, p_targets = draw(probe_rng, PROBE_BATCH)
    p_mask = masked(probe_rng, PROBE_BATCH)

    def probe() -> float:
        model.eval()
        with torch.no_grad():
            loss = floor_weighted_mse(model(p_windows, p_throttle, mask=p_mask), p_targets, floor).item()
        model.train()
        return loss

    losses = [probe()]
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for step in range(steps):
        windows, throttle_hist, targets = draw(sample_rng, BATCH)
        mask = masked(mask_rng, BATCH)
        loss = floor_weighted_mse(model(windows, throttle_hist, mask=mask), targets, floor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % LOG_EVERY == 0:
            losses.append(probe())

    config = {
        "horizon": horizon,
        "mask_prob": mask_prob,
        "seed": seed,
        "steps": steps,
        "n_train_runs": N_TRAIN_RUNS,
        "batch": BATCH,
        "lr": LR,
        "log_every": LOG_EVERY,
        "probe_batch": PROBE_BATCH,
        **architecture(command_input),
        # Recorded, not checked on load. Scoring can pick either, since training never uses it
        "aggregate": AGGREGATE,
    }
    seconds = time.perf_counter() - began
    return TrainResult(model=model.eval(), normalizer=normalizer, losses=losses, config=config, seconds=seconds)


def run_name(config: dict[str, Any]) -> str:
    """Folder name that says which settings made the model. :g writes 0 and 0.25, not 0.0."""
    # .get, since D6 configs have no such key and keep their names. New runs get new folders, so D6's weights survive
    cmd = "_cmd" if config.get("command_aware") else ""
    # D7 configs have no command_input, and all of them read the throttle
    if cmd and not config.get("command_input", True):
        cmd = "_nocmd"
    return f"h{config['horizon']}_mask{config['mask_prob']:g}{cmd}_seed{config['seed']}"


def save_run(result: TrainResult, folder: Path) -> None:
    """config.json and metrics.json go in git. model.pt stays local."""
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), folder / "model.pt")

    # Committed with the config, so anyone can see which statistics a result used
    config = {**result.config, "normalizer": result.normalizer.to_dict()}
    metrics = {"seconds": round(result.seconds, 1), "losses": result.losses}
    (folder / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


def load_run(folder: Path) -> TrainResult:
    """A saved run, ready for scoring. Its config sets the horizon, so scoring can't use the wrong one."""
    config = json.loads((folder / "config.json").read_text())
    # D7's committed configs predate command_input, and every one of them read the throttle
    command_input = config.get("command_input", True)
    # A changed head count keeps every weight shape, so the load would succeed and the model would quietly differ
    expected = architecture(command_input)
    changed = {k: (config.get(k), v) for k, v in expected.items() if k != "command_input" and config.get(k) != v}
    if changed:
        raise ValueError(f"{folder} was trained with different settings (saved, current): {changed}")

    # Built from the saved config, so an ablation always loads as the ablation
    model = Forecaster(config["horizon"], command_input)
    # weights_only refuses anything but tensors, the safe way to read a model file
    model.load_state_dict(torch.load(folder / "model.pt", weights_only=True))
    metrics = json.loads((folder / "metrics.json").read_text())
    normalizer = Normalizer.from_dict(config.pop("normalizer"))
    return TrainResult(
        model=model.eval(), normalizer=normalizer, losses=metrics["losses"], config=config, seconds=metrics["seconds"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the forecaster and save it to runs/.")
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--mask-prob", type=float, default=MASK_PROB)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--no-command", action="store_true", help="the D12 ablation: an all-zero throttle history")
    args = parser.parse_args()

    result = train(args.horizon, args.mask_prob, args.seed, args.steps, command_input=not args.no_command)
    out = RUNS_DIR / run_name(result.config)
    save_run(result, out)
    print(f"saved to {out} in {result.seconds:.0f} s. Probe loss {result.losses[0]:.3f} -> {result.losses[-1]:.3f}")
