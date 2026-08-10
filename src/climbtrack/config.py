"""Strict YAML configuration loading."""

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

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


class ModelsConfig(StrictModel):
    """Model identities; model dependencies are installed in later milestones."""

    primary_pose_backend: str = Field(pattern="^sapiens2$")
    sapiens2: Sapiens2Config


class AppConfig(StrictModel):
    """Root application configuration."""

    project: ProjectConfig
    ingest: IngestConfig
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
