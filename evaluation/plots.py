"""Plot one healthy and one faulted run from the report split, against redlines and the fault window.

Run with: python -m evaluation.plots
PNGs go to runs/plots/, which git ignores.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write image files, never open a window
import matplotlib.pyplot as plt
import numpy as np

from contracts import CHANNELS, DT_S, N_STEPS, Run
from evaluation.budget import CATCH_WINDOW_S, REPORT_SPLIT
from generator.engine import REDLINES, make_throttle, simulate
from generator.splits import SPLITS, fault_for, make_run

OUT = Path("runs/plots")


def plot_run(run: Run, title: str, path: Path, twin: Run | None = None) -> None:
    """Throttle on top, then each channel with its redline. A faulted run shows onset and catch window."""
    t = np.arange(N_STEPS) * DT_S
    fig, axes = plt.subplots(len(CHANNELS) + 1, 1, sharex=True, figsize=(10, 14))
    axes[0].plot(t, run.throttle, color="black", lw=1)
    axes[0].set_ylabel("throttle", rotation=0, ha="right")

    for ax, (i, name) in zip(axes[1:], enumerate(CHANNELS)):
        if twin is not None:
            ax.plot(t, twin.sensors[:, i], color="grey", lw=0.6, label="healthy twin")
        ax.plot(t, run.sensors[:, i], color="tab:blue", lw=0.6, label="this run")
        ax.axhline(REDLINES[name], color="red", ls="--", lw=0.8, label="redline")
        ax.set_ylabel(name, rotation=0, ha="right", fontsize=8)

    if run.fault is not None:
        for ax in axes:
            ax.axvline(run.fault.onset_s, color="orange", lw=1)
            ax.axvspan(run.fault.onset_s, run.fault.onset_s + CATCH_WINDOW_S, color="orange", alpha=0.1)
        axes[1].legend(loc="upper left", fontsize=7)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = SPLITS[REPORT_SPLIT]
    healthy_seed = next(s for s in seeds if fault_for(REPORT_SPLIT, s) is None)
    faulted_seed = next(s for s in seeds if fault_for(REPORT_SPLIT, s) is not None)

    healthy = make_run(REPORT_SPLIT, healthy_seed)
    plot_run(healthy, f"{REPORT_SPLIT} seed {healthy_seed}: healthy", OUT / "healthy.png")

    faulted = make_run(REPORT_SPLIT, faulted_seed)
    twin = simulate(make_throttle(faulted_seed), faulted_seed)  # same engine and throttle, no fault
    onset = faulted.fault.onset_s
    plot_run(faulted, f"{REPORT_SPLIT} seed {faulted_seed}: bearing wear from {onset:.1f} s", OUT / "faulted.png", twin)
    print(f"wrote {OUT / 'healthy.png'} and {OUT / 'faulted.png'}")


if __name__ == "__main__":
    main()
