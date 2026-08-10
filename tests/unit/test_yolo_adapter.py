from pathlib import Path
from types import SimpleNamespace

from climbtrack.backends.yolo11 import Yolo11PersonDetector
from climbtrack.config import DetectionConfig


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def predict(self, *, source, **_kwargs):
        self.calls.append(source)
        return iter(SimpleNamespace(boxes=None) for _ in source)


def test_predict_chunks_paths_before_calling_ultralytics() -> None:
    detector = object.__new__(Yolo11PersonDetector)
    detector.device = "cpu"
    detector.config = DetectionConfig(batch_size=1)
    detector.model = FakeModel()
    paths = [Path(f"frame-{index}.png") for index in range(3)]

    results = list(detector.predict(paths))

    assert results == [(), (), ()]
    assert detector.model.calls == [[str(path)] for path in paths]
