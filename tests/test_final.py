from pathlib import Path

import pytest

from evaluation.final import claim_test_run


def test_claim_writes_the_started_marker(tmp_path: Path) -> None:
    out = tmp_path / "test"
    claim_test_run(out)
    assert (out / "started").exists()


def test_a_second_claim_is_refused(tmp_path: Path) -> None:
    # The test split is opened once. A crash leaves the marker in place, so a rerun needs a logged, manual delete
    out = tmp_path / "test"
    claim_test_run(out)
    with pytest.raises(FileExistsError):
        claim_test_run(out)


def test_a_claim_is_refused_if_results_exist_without_the_marker(tmp_path: Path) -> None:
    # Results saved by any other route still count as the test split having been opened
    out = tmp_path / "test"
    out.mkdir()
    (out / "metrics.json").write_text("{}\n")
    with pytest.raises(FileExistsError):
        claim_test_run(out)
