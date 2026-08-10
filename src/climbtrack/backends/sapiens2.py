"""Official Transformers adapter for pinned Sapiens2-1B pose inference."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from climbtrack.config import Device, PoseConfig
from climbtrack.errors import ClimbTrackError
from climbtrack.schema.keypoints import flip_pairs, read_registry


@dataclass(frozen=True)
class PosePrediction:
    """One raw 308-keypoint prediction in original-image coordinates."""

    coordinates: np.ndarray
    confidence: np.ndarray


class Sapiens2Backend:
    """Pinned local Sapiens2 model with flip and multi-scale TTA."""

    source_backend = "sapiens2"

    def __init__(self, model_dir: Path, *, device: str, config: PoseConfig) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, Sapiens2ForPoseEstimation
        except ImportError as exc:
            raise ClimbTrackError(
                "Sapiens2 dependencies are missing; run 'uv sync --locked'."
            ) from exc

        self._torch = torch
        self._device = device
        self._config = config
        self.registry = read_registry(model_dir / "keypoints.json")
        try:
            self._processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
            dtype = torch.float16 if config.half_precision else torch.float32
            self._model = Sapiens2ForPoseEstimation.from_pretrained(
                model_dir,
                local_files_only=True,
                dtype=dtype,
            ).eval()
            self._model.to(device)
        except Exception as exc:
            raise ClimbTrackError(
                f"Could not load pinned Sapiens2-1B from {model_dir} on {device}: {exc}. "
                "No fallback was used."
            ) from exc

        configured_pairs = sorted(tuple(pair) for pair in self._model.config.flip_pairs)
        registry_pairs = sorted(tuple(pair) for pair in flip_pairs(self.registry))
        if configured_pairs != registry_pairs:
            raise ClimbTrackError(
                "Sapiens2 checkpoint flip pairs do not match the canonical keypoint registry"
            )
        self._flip_pairs = torch.tensor(registry_pairs, dtype=torch.long, device=device)

    def infer_batch(
        self, images_bgr: list[np.ndarray], boxes_xywh: list[list[float]]
    ) -> list[PosePrediction]:
        """Infer a frame batch and average decoded results across configured scales."""
        if len(images_bgr) != len(boxes_xywh) or not images_bgr:
            raise ValueError("images and boxes must be non-empty and have equal length")
        images_rgb = [np.ascontiguousarray(image[:, :, ::-1]) for image in images_bgr]
        nested_boxes = [[box] for box in boxes_xywh]
        scale_predictions: list[list[PosePrediction]] = []
        for scale in self._config.multi_scale_tta:
            height = _scaled_dimension(1024, scale)
            width = _scaled_dimension(768, scale)
            inputs = self._processor(
                images=images_rgb,
                boxes=nested_boxes,
                size={"height": height, "width": width},
                return_tensors="pt",
            )
            pixel_values = inputs["pixel_values"].to(self._device)
            try:
                with self._torch.inference_mode():
                    outputs = self._model(pixel_values=pixel_values)
                    flipped = None
                    if self._config.flip_tta:
                        flipped = self._model(
                            pixel_values=self._torch.flip(pixel_values, dims=(-1,)),
                            flip_pairs=self._flip_pairs,
                        )
                outputs = _output_on_cpu(outputs)
                flipped = _output_on_cpu(flipped) if flipped is not None else None
                decoded = self._processor.post_process_pose_estimation(
                    outputs,
                    boxes=nested_boxes,
                    outputs_flipped=flipped,
                    kernel_size=self._config.postprocess_kernel_size,
                )
            except Exception as exc:
                raise ClimbTrackError(
                    f"Sapiens2 inference failed on {self._device} at {height}x{width}: {exc}. "
                    "No CPU or smaller-model fallback was used."
                ) from exc
            finally:
                del pixel_values
                if self._device == Device.MPS.value:
                    self._torch.mps.empty_cache()
            scale_predictions.append([_decode_person(result) for result in decoded])
        return _average_scales(scale_predictions)


def _scaled_dimension(base: int, scale: float) -> int:
    return max(16, round(base * scale / 16) * 16)


def _output_on_cpu(output: Any) -> Any:
    return type(output)(heatmaps=output.heatmaps.detach().float().cpu())


def _decode_person(image_result: list[dict[str, Any]]) -> PosePrediction:
    if len(image_result) != 1:
        raise ClimbTrackError(
            f"Expected one Sapiens2 person prediction, received {len(image_result)}"
        )
    person = image_result[0]
    coordinates = person["keypoints"].detach().float().cpu().numpy()
    confidence = person["scores"].detach().float().cpu().numpy()
    if coordinates.shape != (308, 2) or confidence.shape != (308,):
        raise ClimbTrackError(
            f"Unexpected Sapiens2 output shapes: {coordinates.shape}, {confidence.shape}"
        )
    if not np.isfinite(coordinates).all() or not np.isfinite(confidence).all():
        raise ClimbTrackError("Sapiens2 produced non-finite keypoints or confidence values")
    return PosePrediction(coordinates=coordinates, confidence=confidence)


def _average_scales(predictions: list[list[PosePrediction]]) -> list[PosePrediction]:
    batch_size = len(predictions[0])
    if any(len(batch) != batch_size for batch in predictions):
        raise ClimbTrackError("Multi-scale Sapiens2 predictions have inconsistent batch sizes")
    averaged = []
    for item in range(batch_size):
        coordinates = np.mean([batch[item].coordinates for batch in predictions], axis=0)
        confidence = np.mean([batch[item].confidence for batch in predictions], axis=0)
        averaged.append(
            PosePrediction(
                coordinates=coordinates.astype(np.float32),
                confidence=confidence.astype(np.float32),
            )
        )
    return averaged
