"""Strict YAML configuration loading."""

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from climbtrack.errors import ConfigurationError


class StrictModel(BaseModel):
    """Base model that rejects unknown configuration keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Device(StrEnum):
    """Explicitly supported compute devices."""

    MPS = "mps"
    CPU = "cpu"
    CUDA = "cuda"


class HdrPolicy(StrEnum):
    """How ingest handles HDR inputs before 8-bit model inference."""

    FAIL = "fail"
    CLIP = "clip"
    TONEMAP = "tonemap"


class ProjectConfig(StrictModel):
    """Project-wide reproducibility settings."""

    cache_dir: Path = Path("cache")
    seed: int = Field(default=42, ge=0)
    device: Device


class IngestConfig(StrictModel):
    """Configuration that affects decoded frame artifacts."""

    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    frame_format: str = Field(default="png", pattern="^png$")
    png_compression: int = Field(default=3, ge=0, le=9)
    hdr_policy: HdrPolicy = HdrPolicy.FAIL
    verify_cached_checksums: bool = True


class Sapiens2Config(StrictModel):
    """Pinned identity for the approved primary pose model."""

    model_id: str
    checkpoint_filename: str
    revision: str | None = None


class Yolo11Config(StrictModel):
    """Pinned identity and explicit download source for YOLO11."""

    model_id: str = Field(pattern="^yolo11x$")
    checkpoint_filename: str = Field(pattern=r"^yolo11x\.pt$")
    source_url: str


class ModelsConfig(StrictModel):
    """Model identities; model dependencies are installed in later milestones."""

    primary_pose_backend: str = Field(pattern="^sapiens2$")
    yolo11: Yolo11Config
    sapiens2: Sapiens2Config


class DetectionConfig(StrictModel):
    """Quality-first YOLO11 person-detection settings."""

    model_path: Path = Path("models/yolo11x.pt")
    image_size: int = Field(default=1280, ge=640, le=2560, multiple_of=32)
    confidence_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.70, gt=0.0, le=1.0)
    max_detections: int = Field(default=50, ge=1, le=1000)
    batch_size: int = Field(default=1, ge=1, le=64)
    half_precision: bool = False


class TrackingConfig(StrictModel):
    """Pinned ByteTrack association settings."""

    track_high_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    track_low_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    new_track_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    track_buffer: int = Field(default=90, ge=1, le=10_000)
    match_threshold: float = Field(default=0.80, gt=0.0, le=1.0)
    fuse_score: bool = True
    containment_threshold: float = Field(default=0.90, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_threshold_order(self) -> "TrackingConfig":
        """Ensure the low-confidence rescue range is meaningful."""
        if self.track_low_threshold >= self.track_high_threshold:
            raise ValueError("track_low_threshold must be below track_high_threshold")
        return self


class SelectionWeights(StrictModel):
    """Relative contribution of independent climber-selection signals."""

    length: float = Field(default=0.25, ge=0.0)
    continuity: float = Field(default=0.20, ge=0.0)
    vertical_range: float = Field(default=0.15, ge=0.0)
    motion: float = Field(default=0.15, ge=0.0)
    center: float = Field(default=0.15, ge=0.0)
    image_area: float = Field(default=0.10, ge=0.0)

    @model_validator(mode="after")
    def require_positive_sum(self) -> "SelectionWeights":
        """Reject a selector with no usable signal."""
        if sum(self.model_dump().values()) <= 0:
            raise ValueError("selection weights must have a positive sum")
        return self


class SelectionConfig(StrictModel):
    """Automatic-selection acceptance thresholds."""

    minimum_observations: int = Field(default=30, ge=2)
    minimum_continuity: float = Field(default=0.45, ge=0.0, le=1.0)
    minimum_score_margin: float = Field(default=0.08, ge=0.0, le=1.0)
    weights: SelectionWeights = SelectionWeights()


class PoseCropConfig(StrictModel):
    """Stable square crop derived from the selected raw tracking box."""

    padding_scale: float = Field(default=1.55, ge=1.0, le=3.0)
    smoothing_window: int = Field(default=15, ge=1, le=121)
    maximum_interpolation_gap: int = Field(default=5, ge=0, le=120)

    @model_validator(mode="after")
    def require_odd_window(self) -> "PoseCropConfig":
        """A centered temporal window must have an unambiguous middle frame."""
        if self.smoothing_window % 2 == 0:
            raise ValueError("pose_crop.smoothing_window must be odd")
        return self


class RenderConfig(StrictModel):
    """Tracking quality-control video settings."""

    ffmpeg_path: str = "ffmpeg"
    codec: str = Field(default="libx264", pattern="^libx264$")
    crf: int = Field(default=18, ge=0, le=51)
    preset: str = Field(default="slow", pattern="^(medium|slow|slower)$")
    line_thickness: int = Field(default=6, ge=1, le=30)
    font_scale: float = Field(default=1.2, gt=0.0, le=5.0)
    show_all_tracks: bool = True


class AppConfig(StrictModel):
    """Root application configuration."""

    project: ProjectConfig
    ingest: IngestConfig
    detection: DetectionConfig
    tracking: TrackingConfig
    selection: SelectionConfig
    pose_crop: PoseCropConfig = PoseCropConfig()
    render: RenderConfig
    models: ModelsConfig


def load_config(path: Path) -> AppConfig:
    """Load and strictly validate a YAML configuration file."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Configuration file does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigurationError(f"Configuration root must be a mapping: {path}")
    try:
        return AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid configuration in {path}:\n{exc}") from exc


def resolve_cache_dir(config: AppConfig, config_path: Path) -> Path:
    """Resolve a relative cache path against the project containing the config."""
    cache_dir = config.project.cache_dir
    if cache_dir.is_absolute():
        return cache_dir
    project_root = config_path.resolve().parent.parent
    return project_root / cache_dir


def resolve_project_path(path: Path, config_path: Path) -> Path:
    """Resolve a configured project-relative path."""
    if path.is_absolute():
        return path.expanduser().resolve()
    project_root = config_path.resolve().parent.parent
    return (project_root / path).resolve()
