"""Loopback-only HTTP server for move playback and manual boundary annotation."""

import errno
import json
import mimetypes
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import TypeAdapter, ValidationError

from climbtrack.config import MovePlayerConfig
from climbtrack.errors import ClimbTrackError
from climbtrack.moves import (
    MoveEdit,
    apply_move_edits,
    load_move_session,
    save_move_session,
)

MAX_REQUEST_BYTES = 1_000_000
MOVE_EDITS = TypeAdapter(list[MoveEdit])


@dataclass(frozen=True)
class PlayerServer:
    """Bound loopback server plus its local URL."""

    httpd: ThreadingHTTPServer
    url: str


@dataclass(frozen=True)
class _PlayerState:
    video_path: Path
    session_path: Path
    frames: list[dict[str, Any]]
    config: MovePlayerConfig
    move_metrics: tuple[dict[str, Any], ...]
    speed_timeline: tuple[dict[str, Any], ...]
    metrics_revision: int
    lock: threading.Lock


def create_player_server(
    video_path: Path,
    session_path: Path,
    frames: list[dict[str, Any]],
    config: MovePlayerConfig,
    *,
    move_metrics: list[dict[str, Any]] | None = None,
    speed_timeline: list[dict[str, Any]] | None = None,
    port: int | None = None,
) -> PlayerServer:
    """Bind a local player server without starting its request loop."""
    source = video_path.expanduser().resolve()
    if not source.is_file():
        raise ClimbTrackError(f"Player video does not exist: {source}")
    if len(frames) < 2:
        raise ClimbTrackError("Player requires at least two source frames")
    state = _PlayerState(
        video_path=source,
        session_path=session_path.resolve(),
        frames=frames,
        config=config,
        move_metrics=tuple(move_metrics or []),
        speed_timeline=tuple(speed_timeline or []),
        metrics_revision=load_move_session(session_path).revision,
        lock=threading.Lock(),
    )
    handler = _handler_for(state)
    preferred_port = config.port if port is None else port
    candidates = (
        [preferred_port]
        if port is not None
        else range(preferred_port, min(preferred_port + 20, 65_535) + 1)
    )
    last_error: OSError | None = None
    for candidate in candidates:
        try:
            httpd = ThreadingHTTPServer((config.host, candidate), handler)
            break
        except OSError as exc:
            last_error = exc
            if exc.errno != errno.EADDRINUSE or port is not None:
                raise ClimbTrackError(
                    f"Move player could not bind to {config.host}:{candidate}: {exc}"
                ) from exc
    else:
        raise ClimbTrackError(
            f"Move player found no free local port from {preferred_port} to "
            f"{min(preferred_port + 20, 65_535)}: {last_error}"
        ) from last_error
    actual_port = int(httpd.server_address[1])
    return PlayerServer(httpd=httpd, url=f"http://{config.host}:{actual_port}/")


def run_player_server(server: PlayerServer, *, open_browser: bool) -> None:
    """Open the player if requested and serve until interrupted."""
    if open_browser:
        webbrowser.open(server.url)
    try:
        server.httpd.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.httpd.server_close()


def parse_byte_range(value: str | None, size: int) -> tuple[int, int] | None:
    """Parse one RFC 7233 byte range; return inclusive bounds."""
    if value is None:
        return None
    if not value.startswith("bytes=") or "," in value:
        raise ValueError("Only one byte range is supported")
    start_text, separator, end_text = value[6:].partition("-")
    if not separator or (not start_text and not end_text):
        raise ValueError("Invalid byte range")
    if start_text:
        start = int(start_text)
        end = int(end_text) if end_text else size - 1
    else:
        suffix = int(end_text)
        if suffix <= 0:
            raise ValueError("Invalid suffix range")
        start = max(0, size - suffix)
        end = size - 1
    if start < 0 or start >= size or end < start:
        raise ValueError("Unsatisfiable byte range")
    return start, min(end, size - 1)


def _handler_for(state: _PlayerState) -> type[BaseHTTPRequestHandler]:
    static_root = Path(__file__).with_name("static")

    class PlayerHandler(BaseHTTPRequestHandler):
        server_version = "ClimbTrackPlayer/1.0"

        def do_GET(self) -> None:
            request = urlsplit(self.path)
            if request.path == "/":
                self._send_file(static_root / "index.html", "text/html; charset=utf-8")
            elif request.path == "/assets/app.js":
                self._send_file(static_root / "app.js", "text/javascript; charset=utf-8")
            elif request.path == "/assets/styles.css":
                self._send_file(static_root / "styles.css", "text/css; charset=utf-8")
            elif request.path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
            elif request.path == "/api/session":
                self._send_session()
            elif request.path == "/video":
                self._send_video(head_only=False)
            elif request.path == "/health":
                self._send_json({"status": "ok"})
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_HEAD(self) -> None:
            request = urlsplit(self.path)
            if request.path == "/video":
                self._send_video(head_only=True)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_PUT(self) -> None:
            request = urlsplit(self.path)
            if request.path != "/api/moves":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._save_moves()

        def _send_session(self) -> None:
            with state.lock:
                session = load_move_session(state.session_path)
            first_timestamp = float(state.frames[0]["timestamp"])
            timeline = [
                {
                    "frame_idx": int(frame["frame_idx"]),
                    "timestamp": float(frame["timestamp"]),
                    "media_time": max(0.0, float(frame["timestamp"]) - first_timestamp),
                    "duration": (None if frame["duration"] is None else float(frame["duration"])),
                }
                for frame in state.frames
            ]
            self._send_json(
                {
                    "session": session.model_dump(mode="json"),
                    "metrics": (
                        list(state.move_metrics)
                        if session.revision == state.metrics_revision
                        else []
                    ),
                    "speed_timeline": (
                        list(state.speed_timeline)
                        if session.revision == state.metrics_revision
                        else []
                    ),
                    "timeline": timeline,
                    "video": {
                        "name": state.video_path.name,
                        "url": "/video",
                    },
                    "settings": {
                        "lead_in_seconds": state.config.lead_in_seconds,
                        "lead_out_seconds": state.config.lead_out_seconds,
                    },
                }
            )

        def _save_moves(self) -> None:
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self.send_error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Expected application/json")
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
                return
            if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
                self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                return
            try:
                payload = json.loads(self.rfile.read(content_length))
                expected_revision = int(payload["expected_revision"])
                edits = MOVE_EDITS.validate_python(payload["moves"])
                with state.lock:
                    current = load_move_session(state.session_path)
                    updated = apply_move_edits(
                        current,
                        edits,
                        state.frames,
                        expected_revision=expected_revision,
                    )
                    save_move_session(updated, state.session_path)
            except ClimbTrackError as exc:
                status = (
                    HTTPStatus.CONFLICT
                    if "another browser tab" in str(exc)
                    else HTTPStatus.BAD_REQUEST
                )
                self._send_json({"error": str(exc)}, status=status)
                return
            except (KeyError, TypeError, ValueError, json.JSONDecodeError, ValidationError) as exc:
                self._send_json(
                    {"error": f"Invalid move payload: {exc}"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            self._send_json({"session": updated.model_dump(mode="json")})

        def _send_video(self, *, head_only: bool) -> None:
            size = state.video_path.stat().st_size
            try:
                byte_range = parse_byte_range(self.headers.get("Range"), size)
            except (ValueError, TypeError):
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            start, end = byte_range if byte_range is not None else (0, size - 1)
            length = end - start + 1
            self.send_response(HTTPStatus.PARTIAL_CONTENT if byte_range else HTTPStatus.OK)
            self._security_headers()
            content_type = mimetypes.guess_type(state.video_path.name)[0] or "video/mp4"
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if byte_range:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if head_only:
                return
            with state.video_path.open("rb") as source:
                source.seek(start)
                remaining = length
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    remaining -= len(chunk)

        def _send_file(self, path: Path, content_type: str) -> None:
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            payload = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self._security_headers()
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(
            self,
            payload: dict[str, Any],
            *,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            encoded = json.dumps(payload, separators=(",", ":")).encode()
            self.send_response(status)
            self._security_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

        def _security_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self' data:; media-src 'self'; connect-src 'self'",
            )

        def log_message(self, format: str, *args: object) -> None:
            return

    return PlayerHandler
