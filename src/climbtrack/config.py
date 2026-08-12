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
    annotation_dir: Path = Path("annotations")
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
    model_dir: Path
    checkpoint_filename: str
    checkpoint_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    checkpoint_size_bytes: int = Field(gt=0)
    download_connections: int = Field(default=16, ge=1, le=32)
    download_segment_mb: int = Field(default=8, ge=4, le=256)
    revision: str = Field(pattern="^[0-9a-f]{40}$")
    keypoint_source_url: str


class Yolo11Config(StrictModel):
    """Pinned identity and explicit download source for YOLO11."""

    model_id: str = Field(pattern="^yolo11x$")
    checkpoint_filename: str = Field(pattern=r"^yolo11x\.pt$")
    checkpoint_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    checkpoint_size_bytes: int = Field(gt=0)
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
    """Stable model-aspect crop derived from nearby selected-person boxes."""

    input_width: int = Field(default=768, ge=16, multiple_of=16)
    input_height: int = Field(default=1024, ge=16, multiple_of=16)
    padding_scale: float = Field(default=1.35, ge=1.0, le=3.0)
    context_window: int = Field(default=91, ge=1, le=241)
    smoothing_window: int = Field(default=15, ge=1, le=121)
    maximum_interpolation_gap: int = Field(default=5, ge=0, le=120)

    @model_validator(mode="after")
    def require_odd_window(self) -> "PoseCropConfig":
        """A centered temporal window must have an unambiguous middle frame."""
        if self.context_window % 2 == 0 or self.smoothing_window % 2 == 0:
            raise ValueError("pose_crop temporal windows must be odd")
        return self


class PoseConfig(StrictModel):
    """Quality-first raw pose inference settings."""

    batch_size: int = Field(default=1, ge=1, le=8)
    half_precision: bool = False
    flip_tta: bool = True
    multi_scale_tta: tuple[float, ...] = (1.0, 1.125)
    postprocess_kernel_size: int = Field(default=11, ge=3, le=31)

    @model_validator(mode="after")
    def validate_pose_settings(self) -> "PoseConfig":
        """Require actual multi-scale TTA and an odd refinement kernel."""
        if len(self.multi_scale_tta) < 2:
            raise ValueError("pose.multi_scale_tta must contain at least two scales")
        if any(scale < 0.5 or scale > 1.5 for scale in self.multi_scale_tta):
            raise ValueError("pose.multi_scale_tta scales must be in [0.5, 1.5]")
        if len(set(self.multi_scale_tta)) != len(self.multi_scale_tta):
            raise ValueError("pose.multi_scale_tta scales must be unique")
        if self.postprocess_kernel_size % 2 == 0:
            raise ValueError("pose.postprocess_kernel_size must be odd")
        return self


class PoseRenderConfig(StrictModel):
    """Raw skeleton quality-control rendering settings."""

    confidence_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    point_radius: int = Field(default=5, ge=1, le=30)
    line_thickness: int = Field(default=5, ge=1, le=30)
    show_face_keypoints: bool = False
    show_pose_crop: bool = False
    comparison_panel_width: int = Field(default=1080, ge=320, le=2160, multiple_of=2)


class AnnotationConfig(StrictModel):
    """Small, difficult-frame ground-truth workflow settings."""

    sample_count: int = Field(default=10, ge=3, le=100)
    minimum_spacing_seconds: float = Field(default=0.35, ge=0.0, le=30.0)
    confidence_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    pck_threshold: float = Field(default=0.20, gt=0.0, le=1.0)
    oks_sigma: float = Field(default=0.10, gt=0.0, le=1.0)


class RefineConfig(StrictModel):
    """Conservative temporal pose-repair settings."""

    confidence_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    confidence_threshold_overrides: dict[str, float] = Field(
        default_factory=lambda: {
            "body": 0.05,
            "extra": 0.05,
            "feet": 0.05,
            "left_hand": 0.08,
            "right_hand": 0.08,
        }
    )
    smoothing_groups: tuple[str, ...] = ("left_hand", "right_hand")
    maximum_interpolation_gap: int = Field(default=5, ge=0, le=60)
    one_euro_min_cutoff: float = Field(default=12.0, gt=0.0, le=30.0)
    one_euro_beta: float = Field(default=0.03, ge=0.0, le=5.0)
    one_euro_derivative_cutoff: float = Field(default=1.0, gt=0.0, le=30.0)
    segment_maximum_ratio: float = Field(default=2.5, gt=1.0, le=10.0)
    outlier_confidence_ceiling: float = Field(default=0.50, ge=0.0, le=1.0)
    swap_cost_ratio: float = Field(default=0.35, gt=0.0, lt=1.0)
    swap_minimum_jump_scale: float = Field(default=0.20, ge=0.0, le=2.0)

    @model_validator(mode="after")
    def validate_group_settings(self) -> "RefineConfig":
        """Limit group-specific behavior to canonical Sapiens groups."""
        allowed = {"body", "extra", "face", "feet", "left_hand", "right_hand"}
        unknown = (
            self.confidence_threshold_overrides.keys() | set(self.smoothing_groups)
        ) - allowed
        if unknown:
            raise ValueError(f"Unknown refinement groups: {sorted(unknown)}")
        if any(
            value < 0.0 or value > 1.0 for value in self.confidence_threshold_overrides.values()
        ):
            raise ValueError("Group confidence thresholds must be in [0, 1]")
        if len(set(self.smoothing_groups)) != len(self.smoothing_groups):
            raise ValueError("refine.smoothing_groups must be unique")
        return self


class MovePlayerConfig(StrictModel):
    """Local-only Phase-2 move player settings."""

    host: str = Field(default="127.0.0.1", pattern=r"^127\.0\.0\.1$")
    port: int = Field(default=8765, ge=1024, le=65_535)
    lead_in_seconds: float = Field(default=0.25, ge=0.0, le=2.0)
    lead_out_seconds: float = Field(default=0.25, ge=0.0, le=2.0)


class MoveDetectionConfig(StrictModel):
    """Scale-normalized automatic hand-move segmentation settings."""

    position_smoothing_radius: int = Field(default=4, ge=1, le=30)
    speed_window_radius: int = Field(default=7, ge=1, le=60)
    stable_speed_body_lengths_per_second: float = Field(default=0.18, gt=0.0, le=2.0)
    start_speed_body_lengths_per_second: float = Field(default=0.45, gt=0.0, le=5.0)
    minimum_stable_seconds: float = Field(default=0.50, ge=0.1, le=5.0)
    maximum_stable_gap_seconds: float = Field(default=0.18, ge=0.0, le=1.0)
    same_hold_radius_body_lengths: float = Field(default=0.12, gt=0.0, le=1.0)
    minimum_displacement_body_lengths: float = Field(default=0.25, gt=0.0, le=2.0)
    minimum_move_seconds: float = Field(default=0.10, gt=0.0, le=2.0)
    maximum_move_seconds: float = Field(default=6.0, gt=0.1, le=30.0)
    body_motion_quantile: float = Field(default=0.70, ge=0.5, le=1.0)
    body_stable_speed_body_lengths_per_second: float = Field(default=0.40, gt=0.0, le=5.0)
    minimum_body_stable_seconds: float = Field(default=0.50, ge=0.1, le=5.0)
    fall_minimum_drop_body_lengths: float = Field(default=0.50, gt=0.0, le=5.0)


class MoveMetricsConfig(StrictModel):
    """Smoothing and quality thresholds for per-move kinematics."""

    position_smoothing_radius: int = Field(default=4, ge=1, le=30)
    speed_window_radius: int = Field(default=2, ge=1, le=60)
    body_length_smoothing_radius: int = Field(default=30, ge=1, le=600)
    minimum_valid_fraction: float = Field(default=0.80, ge=0.5, le=1.0)
    settle_speed_body_lengths_per_second: float = Field(default=0.18, ge=0.01, le=5.0)
    coordination_maximum_lag_seconds: float = Field(default=1.0, ge=0.05, le=5.0)


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
    pose: PoseConfig = PoseConfig()
    render: RenderConfig
    pose_render: PoseRenderConfig = PoseRenderConfig()
    annotation: AnnotationConfig = AnnotationConfig()
    refine: RefineConfig = RefineConfig()
    move_player: MovePlayerConfig = MovePlayerConfig()
    move_detection: MoveDetectionConfig = MoveDetectionConfig()
    move_metrics: MoveMetricsConfig = MoveMetricsConfig()
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


def resolve_annotation_dir(config: AppConfig, config_path: Path) -> Path:
    """Resolve the editable ground-truth directory against the project root."""
    annotation_dir = config.project.annotation_dir
    if annotation_dir.is_absolute():
        return annotation_dir
    project_root = config_path.resolve().parent.parent
    return project_root / annotation_dir


def resolve_project_path(path: Path, config_path: Path) -> Path:
    """Resolve a configured project-relative path."""
    if path.is_absolute():
        return path.expanduser().resolve()
    project_root = config_path.resolve().parent.parent
    return (project_root / path).resolve()
