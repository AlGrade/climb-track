import json
import threading
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import pytest

from climbtrack.config import MovePlayerConfig
from climbtrack.moves import MoveSession, save_move_session
from climbtrack.player.server import create_player_server, parse_byte_range
from climbtrack.schema.moves import read_moves_parquet


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (None, None),
        ("bytes=2-5", (2, 5)),
        ("bytes=7-", (7, 9)),
        ("bytes=-3", (7, 9)),
        ("bytes=0-99", (0, 9)),
    ],
)
def test_parse_byte_range(header: str | None, expected: tuple[int, int] | None) -> None:
    assert parse_byte_range(header, 10) == expected


@pytest.mark.parametrize("header", ["items=0-1", "bytes=", "bytes=20-30", "bytes=4-2"])
def test_parse_byte_range_rejects_invalid_input(header: str) -> None:
    with pytest.raises(ValueError):
        parse_byte_range(header, 10)


def test_player_serves_ranges_and_atomically_saves_moves(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"0123456789")
    session_path = tmp_path / "moves_ground_truth.json"
    save_move_session(
        MoveSession(
            created_at="2026-08-11T00:00:00+00:00",
            updated_at="2026-08-11T00:00:00+00:00",
            source_video_name=video.name,
            ingest_cache_key="ingest",
            frame_count=3,
            first_timestamp=0.0,
            last_timestamp=0.2,
            moves=[],
        ),
        session_path,
    )
    frames = [{"frame_idx": index, "timestamp": index * 0.1, "duration": 0.1} for index in range(3)]
    player = create_player_server(
        video,
        session_path,
        frames,
        MovePlayerConfig(),
        move_metrics=[{"move_id": 1, "hand_max_speed_px_s": 42.0}],
        speed_timeline=[{"move_id": 1, "frame_idx": 0, "hand_speed_px_s": 20.0}],
        port=0,
    )
    thread = threading.Thread(target=player.httpd.serve_forever, daemon=True)
    thread.start()
    parsed = urlsplit(player.url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    try:
        with urlopen(origin) as response:
            player_html = response.read().decode("utf-8")
        assert 'id="videoScrubber"' in player_html
        assert 'id="videoToggle"' in player_html
        assert 'id="layoutToggle"' in player_html

        with urlopen(f"{origin}/assets/app.js") as response:
            player_javascript = response.read().decode("utf-8")
        assert "previewScrub" in player_javascript
        assert "finishFrameStep" in player_javascript
        assert "initializeLayout" in player_javascript

        with urlopen(f"{origin}/api/session") as response:
            payload = json.load(response)
        assert payload["session"]["moves"] == []
        assert payload["timeline"][2]["media_time"] == pytest.approx(0.2)
        assert payload["metrics"][0]["hand_max_speed_px_s"] == 42.0
        assert payload["speed_timeline"][0]["frame_idx"] == 0

        range_request = Request(
            f"{origin}/video",
            headers={"Range": "bytes=2-5"},
        )
        with urlopen(range_request) as response:
            assert response.status == 206
            assert response.read() == b"2345"
            assert response.headers["Content-Range"] == "bytes 2-5/10"

        save_request = Request(
            f"{origin}/api/moves",
            method="PUT",
            headers={
                "Content-Type": "application/json",
            },
            data=json.dumps(
                {
                    "expected_revision": 0,
                    "moves": [
                        {
                            "start_frame": 0,
                            "end_frame": 2,
                            "moving_hand": "left",
                        }
                    ],
                }
            ).encode(),
        )
        with urlopen(save_request) as response:
            saved = json.load(response)
        assert saved["session"]["revision"] == 1
        assert saved["session"]["moves"][0]["start_timestamp"] == 0.0
        assert read_moves_parquet(tmp_path / "moves.parquet")[0]["moving_hand"] == "left"
        with urlopen(f"{origin}/api/session") as response:
            refreshed = json.load(response)
        assert refreshed["metrics"] == []
        assert refreshed["speed_timeline"] == []
    finally:
        player.httpd.shutdown()
        player.httpd.server_close()
        thread.join(timeout=2)
