"""Optional local click-based track selection."""

from collections import defaultdict
from pathlib import Path
from typing import Any

from climbtrack.errors import ConfigurationError


def choose_track_by_click(
    ingest_path: Path,
    frames: list[dict[str, Any]],
    tracks: list[dict[str, Any]],
) -> int:
    """Show a representative frame and return the clicked track ID."""
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in tracks:
        by_frame[int(row["frame_idx"])].append(row)
    if not by_frame:
        raise ConfigurationError("There are no tracks to select")
    middle = len(frames) // 2
    max_people = max(len(rows) for rows in by_frame.values())
    reference_idx = min(
        (index for index, rows in by_frame.items() if len(rows) == max_people),
        key=lambda index: abs(index - middle),
    )
    frame = frames[reference_idx]
    image_path = ingest_path / str(frame["image_path"])
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ConfigurationError(f"Could not load reference frame: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    _, axis = plt.subplots(figsize=(10, 16))
    axis.imshow(image)
    axis.set_title(f"Click the climber — frame {reference_idx}")
    axis.axis("off")
    rows = by_frame[reference_idx]
    for row in rows:
        x1, y1 = float(row["x1"]), float(row["y1"])
        width = float(row["x2"]) - x1
        height = float(row["y2"]) - y1
        axis.add_patch(Rectangle((x1, y1), width, height, fill=False, color="yellow", linewidth=2))
        axis.text(x1, y1, f"ID {row['track_id']}", color="yellow", fontsize=12)
    clicks = plt.ginput(1, timeout=-1)
    plt.close()
    if not clicks:
        raise ConfigurationError("Click selection was cancelled")
    x, y = clicks[0]
    matches = [
        row
        for row in rows
        if float(row["x1"]) <= x <= float(row["x2"]) and float(row["y1"]) <= y <= float(row["y2"])
    ]
    if not matches:
        raise ConfigurationError("The click was outside every tracked person box")
    chosen = min(
        matches,
        key=lambda row: (
            (float(row["x2"]) - float(row["x1"])) * (float(row["y2"]) - float(row["y1"]))
        ),
    )
    return int(chosen["track_id"])
