import json
from pathlib import Path

import pytest

from climbtrack.errors import ClimbTrackError
from climbtrack.moves import MoveEdit, MoveSession, apply_move_edits, save_move_session
from climbtrack.schema.moves import read_moves_parquet


def _empty_session() -> MoveSession:
    return MoveSession(
        created_at="2026-08-11T00:00:00+00:00",
        updated_at="2026-08-11T00:00:00+00:00",
        source_video_name="climb.mp4",
        ingest_cache_key="ingest-key",
        frame_count=4,
        first_timestamp=0.0,
        last_timestamp=0.3,
        moves=[],
    )


def _frames() -> list[dict[str, object]]:
    return [{"frame_idx": index, "timestamp": index * 0.1, "duration": 0.1} for index in range(4)]


def test_move_edits_are_sorted_renumbered_and_snapped_to_frames(tmp_path: Path) -> None:
    updated = apply_move_edits(
        _empty_session(),
        [
            MoveEdit(start_frame=2, end_frame=3, moving_hand="right"),
            MoveEdit(start_frame=0, end_frame=1, moving_hand="left"),
        ],
        _frames(),
        expected_revision=0,
    )

    assert updated.revision == 1
    assert [move.move_id for move in updated.moves] == [1, 2]
    assert [move.moving_hand for move in updated.moves] == ["left", "right"]
    assert updated.moves[1].start_timestamp == pytest.approx(0.2)
    assert updated.moves[1].end_timestamp == pytest.approx(0.3)

    path = tmp_path / "moves_ground_truth.json"
    save_move_session(updated, path)

    assert json.loads(path.read_text(encoding="utf-8"))["revision"] == 1
    rows = read_moves_parquet(tmp_path / "moves.parquet")
    assert [row["moving_hand"] for row in rows] == ["left", "right"]
    assert rows[0]["is_reviewed"] is True
    assert rows[0]["outcome"] == "completed"


def test_move_edits_reject_stale_revision_and_duplicates() -> None:
    session = _empty_session()
    edit = MoveEdit(start_frame=0, end_frame=1, moving_hand="both")

    with pytest.raises(ClimbTrackError, match="another browser tab"):
        apply_move_edits(session, [edit], _frames(), expected_revision=1)

    with pytest.raises(ClimbTrackError, match="Duplicate"):
        apply_move_edits(session, [edit, edit], _frames(), expected_revision=0)
