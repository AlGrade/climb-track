from pathlib import Path

from climbtrack.stages.player_video import player_video_command


def test_player_video_command_builds_short_gop_web_proxy() -> None:
    command = player_video_command(
        Path("/tools/ffmpeg"),
        Path("/cache/skeleton.mp4"),
        Path("/cache/player.mp4"),
    )

    assert command[0] == "/tools/ffmpeg"
    assert "scale=w='min(1080,iw)':h=-2" in command
    assert command[command.index("-g") + 1] == "15"
    assert command[command.index("-keyint_min") + 1] == "15"
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert command[command.index("-fps_mode") + 1] == "vfr"
    assert command[command.index("-movflags") + 1] == "+faststart"
    assert command[-1] == "/cache/player.mp4"
