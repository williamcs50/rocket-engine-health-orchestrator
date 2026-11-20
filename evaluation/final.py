"""The one test run (D9). Decision log, 2026-11-19 (test run reporting) and 2026-11-20 (attribution)."""

from pathlib import Path

# Written before any test run is simulated, so a crash still counts as the test split having been opened
STARTED = "started"
METRICS = "metrics.json"


def claim_test_run(out: Path) -> None:
    """Refuse if this folder already holds a started marker or results, otherwise write the marker."""
    for name in (STARTED, METRICS):
        if (out / name).exists():
            raise FileExistsError(
                f"{out / name} exists, so the test split was already opened. "
                "A rerun needs the marker deleted by hand and the reason logged"
            )
    out.mkdir(parents=True, exist_ok=True)
    # exclusive mode, so two runs started at the same moment can't both claim it
    with open(out / STARTED, "x") as f:
        f.write("claimed\n")
