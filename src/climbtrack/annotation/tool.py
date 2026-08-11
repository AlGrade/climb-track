"""Small resumable matplotlib editor for prediction-initialized ground truth."""

from pathlib import Path
from typing import Any

from climbtrack.annotation.session import AnnotationPoint, AnnotationSession, save_session
from climbtrack.errors import ClimbTrackError

GROUP_COLORS = {
    "body": "#00e676",
    "feet": "#ffd600",
    "extra": "#ff4081",
    "left_hand": "#40c4ff",
    "right_hand": "#7c4dff",
}


def launch_annotation_tool(
    session: AnnotationSession,
    session_path: Path,
    ingest_path: Path,
    skeleton_edges: list[list[str]],
) -> None:
    """Open the local point editor and block until its window is closed."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    editor = _AnnotationEditor(session, session_path, ingest_path, skeleton_edges, plt)
    previous_axis = plt.axes((0.05, 0.02, 0.13, 0.055))
    approve_axis = plt.axes((0.20, 0.02, 0.24, 0.055))
    missing_axis = plt.axes((0.46, 0.02, 0.16, 0.055))
    reset_axis = plt.axes((0.64, 0.02, 0.16, 0.055))
    close_axis = plt.axes((0.82, 0.02, 0.13, 0.055))
    buttons = (
        Button(previous_axis, "Zurück"),
        Button(approve_axis, "Bestätigen + weiter"),
        Button(missing_axis, "Unsichtbar"),
        Button(reset_axis, "Zurücksetzen"),
        Button(close_axis, "Schließen"),
    )
    buttons[0].on_clicked(editor.previous)
    buttons[1].on_clicked(editor.approve_and_next)
    buttons[2].on_clicked(editor.toggle_missing)
    buttons[3].on_clicked(editor.reset_selected)
    buttons[4].on_clicked(editor.close)
    editor.draw()
    plt.show()


class _AnnotationEditor:
    def __init__(
        self,
        session: AnnotationSession,
        session_path: Path,
        ingest_path: Path,
        skeleton_edges: list[list[str]],
        pyplot: Any,
    ) -> None:
        self.session = session
        self.session_path = session_path
        self.ingest_path = ingest_path
        self.skeleton_edges = skeleton_edges
        self.pyplot = pyplot
        self.figure, self.axis = pyplot.subplots(figsize=(9, 10))
        self.figure.subplots_adjust(bottom=0.11, top=0.93, left=0.04, right=0.98)
        self.index = next(
            (index for index, frame in enumerate(session.frames) if not frame.reviewed), 0
        )
        self.selected_name = session.keypoint_names[0]
        self.dragging = False
        self.figure.canvas.mpl_connect("button_press_event", self._press)
        self.figure.canvas.mpl_connect("motion_notify_event", self._motion)
        self.figure.canvas.mpl_connect("button_release_event", self._release)
        self.figure.canvas.mpl_connect("key_press_event", self._key)
        self.figure.canvas.mpl_connect("close_event", self._window_closed)

    @property
    def frame(self) -> Any:
        return self.session.frames[self.index]

    def draw(self) -> None:
        import cv2

        frame = self.frame
        source = self.ingest_path / frame.image_path
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ClimbTrackError(f"Could not read annotation frame: {source}")
        height, width = image.shape[:2]
        x1 = max(0, min(width - 1, round(frame.crop.x1)))
        y1 = max(0, min(height - 1, round(frame.crop.y1)))
        x2 = max(x1 + 1, min(width, round(frame.crop.x2)))
        y2 = max(y1 + 1, min(height, round(frame.crop.y2)))
        rgb = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)

        self.axis.clear()
        self.axis.imshow(rgb, extent=(x1, x2, y2, y1))
        self.axis.set_xlim(x1, x2)
        self.axis.set_ylim(y2, y1)
        self.axis.set_aspect("equal")
        self.axis.axis("off")
        self._draw_edges()
        self._draw_points()
        reviewed = sum(frame.reviewed for frame in self.session.frames)
        state = "bestätigt" if frame.reviewed else "noch offen"
        self.axis.set_title(
            f"Bild {self.index + 1}/{len(self.session.frames)} · {frame.timestamp:.2f} s · "
            f"{state} · {reviewed}/{len(self.session.frames)} fertig\n"
            f"Ausgewählt: {self.selected_name} | Ziehen = korrigieren | Rechtsklick = unsichtbar",
            fontsize=11,
        )
        self.figure.canvas.draw_idle()

    def _draw_edges(self) -> None:
        points = self.frame.points
        for left, right in self.skeleton_edges:
            if left not in points or right not in points:
                continue
            first, second = points[left], points[right]
            if not first.visible or not second.visible:
                continue
            self.axis.plot(
                (first.x, second.x),
                (first.y, second.y),
                color="#ffffff",
                linewidth=1.2,
                alpha=0.65,
                zorder=2,
            )

    def _draw_points(self) -> None:
        for point in self.frame.points.values():
            selected = point.name == self.selected_name
            if point.visible:
                self.axis.scatter(
                    [point.x],
                    [point.y],
                    s=95 if selected else 48,
                    c=[GROUP_COLORS.get(point.group, "#ffffff")],
                    edgecolors="#000000",
                    linewidths=2.0 if selected else 0.8,
                    zorder=4,
                )
            else:
                self.axis.scatter(
                    [point.predicted_x],
                    [point.predicted_y],
                    s=95 if selected else 55,
                    marker="x",
                    c=["#bdbdbd"],
                    linewidths=2.0,
                    zorder=4,
                )

    def _press(self, event: Any) -> None:
        if event.inaxes is not self.axis or event.x is None or event.y is None:
            return
        nearest = self._nearest(event.x, event.y)
        if nearest is None:
            return
        self.selected_name = nearest.name
        if event.button == 3:
            self._toggle(nearest)
            self._persist()
            self.draw()
            return
        if event.button == 1 and nearest.visible:
            self.dragging = True
            self.draw()

    def _motion(self, event: Any) -> None:
        if not self.dragging or event.inaxes is not self.axis:
            return
        if event.xdata is None or event.ydata is None:
            return
        point = self.frame.points[self.selected_name]
        point.x, point.y = float(event.xdata), float(event.ydata)
        self.draw()

    def _release(self, _event: Any) -> None:
        if self.dragging:
            self.dragging = False
            self._persist()

    def _nearest(self, display_x: float, display_y: float) -> AnnotationPoint | None:
        candidates = []
        for point in self.frame.points.values():
            x = point.x if point.visible else point.predicted_x
            y = point.y if point.visible else point.predicted_y
            transformed = self.axis.transData.transform((x, y))
            distance = (
                (transformed[0] - display_x) ** 2 + (transformed[1] - display_y) ** 2
            ) ** 0.5
            candidates.append((distance, point.name, point))
        distance, _, point = min(candidates)
        return point if distance <= 24.0 else None

    def previous(self, _event: Any = None) -> None:
        self.index = (self.index - 1) % len(self.session.frames)
        self.draw()

    def approve_and_next(self, _event: Any = None) -> None:
        self.frame.reviewed = True
        self._persist()
        if all(frame.reviewed for frame in self.session.frames):
            self.close()
            return
        self.index = (self.index + 1) % len(self.session.frames)
        while self.session.frames[self.index].reviewed:
            self.index = (self.index + 1) % len(self.session.frames)
        self.draw()

    def toggle_missing(self, _event: Any = None) -> None:
        self._toggle(self.frame.points[self.selected_name])
        self._persist()
        self.draw()

    def reset_selected(self, _event: Any = None) -> None:
        point = self.frame.points[self.selected_name]
        point.visible = True
        point.x, point.y = point.predicted_x, point.predicted_y
        self._persist()
        self.draw()

    def close(self, _event: Any = None) -> None:
        self._persist()
        self.pyplot.close(self.figure)

    def _key(self, event: Any) -> None:
        if event.key in {"enter", "return"}:
            self.approve_and_next()
        elif event.key == "left":
            self.previous()
        elif event.key == "right":
            self.index = (self.index + 1) % len(self.session.frames)
            self.draw()
        elif event.key == "m":
            self.toggle_missing()
        elif event.key == "r":
            self.reset_selected()

    def _toggle(self, point: AnnotationPoint) -> None:
        point.visible = not point.visible
        if point.visible:
            point.x, point.y = point.predicted_x, point.predicted_y
        else:
            point.x, point.y = None, None

    def _persist(self) -> None:
        save_session(self.session, self.session_path)

    def _window_closed(self, _event: Any) -> None:
        self._persist()
