"""Explicit acquisition of pinned local model files."""

from climbtrack.model_downloads.sapiens2 import (
    ensure_sapiens2_checkpoint,
    verify_sapiens2_checkpoint,
)
from climbtrack.model_downloads.yolo import ensure_yolo11_checkpoint, verify_yolo11_checkpoint

__all__ = [
    "ensure_sapiens2_checkpoint",
    "ensure_yolo11_checkpoint",
    "verify_sapiens2_checkpoint",
    "verify_yolo11_checkpoint",
]
