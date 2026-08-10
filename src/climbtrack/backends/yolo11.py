"""Pinned Ultralytics YOLO11 person-detection adapter."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from typing import Any

from climbtrack.config import DetectionConfig, Device
from climbtrack.device import require_torch_device, seed_torch
from climbtrack.errors import ExternalToolError


@dataclass(frozen=True)
class Detection:
    """One person detection in original-frame pixel coordinates."""

    detection_idx: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


class Yolo11PersonDetector:
    """YOLO11x adapter that never downloads or changes devices implicitly."""

    def __init__(
        self,
        model_path: Path,
        config: DetectionConfig,
        device: Device,
        seed: int,
    ) -> None:
        if not model_path.is_file():
            raise ExternalToolError(
                f"YOLO11 checkpoint is missing: {model_path}. "
                "Download the configured checkpoint explicitly before detection."
            )
        self.device = require_torch_device(device)
        seed_torch(seed)
        from ultralytics import YOLO

        self.model = YOLO(str(model_path), task="detect", verbose=False)
        self.config = config

    def predict(self, image_paths: Iterable[Path]) -> Iterator[tuple[Detection, ...]]:
        """Yield person detections in exactly the input image order."""
        sources = (str(path) for path in image_paths)
        try:
            for source_batch in batched(sources, self.config.batch_size):
                results: Iterable[Any] = self.model.predict(
                    source=list(source_batch),
                    stream=True,
                    imgsz=self.config.image_size,
                    conf=self.config.confidence_threshold,
                    iou=self.config.iou_threshold,
                    max_det=self.config.max_detections,
                    classes=[0],
                    device=self.device,
                    quantize=16 if self.config.half_precision else None,
                    batch=self.config.batch_size,
                    verbose=False,
                    save=False,
                )
                for result in results:
                    boxes = result.boxes
                    if boxes is None or len(boxes) == 0:
                        yield ()
                        continue
                    coordinates = boxes.xyxy.detach().cpu().numpy()
                    confidences = boxes.conf.detach().cpu().numpy()
                    classes = boxes.cls.detach().cpu().numpy()
                    yield tuple(
                        Detection(
                            detection_idx=index,
                            x1=float(coordinates[index, 0]),
                            y1=float(coordinates[index, 1]),
                            x2=float(coordinates[index, 2]),
                            y2=float(coordinates[index, 3]),
                            confidence=float(confidences[index]),
                            class_id=int(classes[index]),
                        )
                        for index in range(len(coordinates))
                    )
        except RuntimeError as exc:
            raise ExternalToolError(
                f"YOLO11 inference failed on device '{self.device}' with "
                f"batch_size={self.config.batch_size}: {exc}"
            ) from exc
