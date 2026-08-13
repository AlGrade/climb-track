import json
import threading
from http import HTTPStatus
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import pytest

from climbtrack.config import MovePlayerConfig
from climbtrack.moves import MoveSession, save_move_session
from climbtrack.player.server import create_player_server, parse_byte_range, resolve_asset
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


def test_resolve_asset_returns_files_inside_the_static_root(tmp_path: Path) -> None:
    (tmp_path / "js").mkdir()
    (tmp_path / "js" / "main.js").write_text("export {};", encoding="utf-8")

    resolved = resolve_asset(tmp_path, "js/main.js")

    assert resolved == (tmp_path / "js" / "main.js").resolve()


@pytest.mark.parametrize(
    "relative",
    [
        "../secret.txt",
        "js/../../secret.txt",
        "..%2fsecret.txt",
        "%2e%2e/secret.txt",
        "/etc/hosts",
        "",
    ],
)
def test_resolve_asset_rejects_paths_outside_the_static_root(tmp_path: Path, relative: str) -> None:
    (tmp_path / "secret.txt").write_text("private", encoding="utf-8")
    static_root = tmp_path / "static"
    static_root.mkdir()

    assert resolve_asset(static_root, relative) is None


def test_resolve_asset_rejects_symlinks_leaving_the_static_root(tmp_path: Path) -> None:
    outside = tmp_path / "secret.txt"
    outside.write_text("private", encoding="utf-8")
    static_root = tmp_path / "static"
    static_root.mkdir()
    (static_root / "escape.js").symlink_to(outside)

    assert resolve_asset(static_root, "escape.js") is None


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
        assert 'id="movesCard"' in player_html
        assert 'id="transport"' in player_html
        assert 'id="editorCard"' in player_html

        with urlopen(f"{origin}/assets/js/main.js") as response:
            assert response.headers["Content-Type"].startswith("text/javascript")
            entry_point = response.read().decode("utf-8")
        assert "./playback.js" in entry_point
        assert "./selection.js" in entry_point

        with urlopen(f"{origin}/assets/js/playback.js") as response:
            playback_module = response.read().decode("utf-8")
        assert "previewScrub" in playback_module
        assert "finishFrameStep" in playback_module

        with urlopen(f"{origin}/assets/styles/base.css") as response:
            assert response.headers["Content-Type"].startswith("text/css")
            assert ":root" in response.read().decode("utf-8")

        for missing in ("/assets/app.js", "/assets/../server.py", "/assets/js/"):
            with pytest.raises(HTTPError) as rejected:
                urlopen(f"{origin}{missing}")
            assert rejected.value.code == HTTPStatus.NOT_FOUND

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
