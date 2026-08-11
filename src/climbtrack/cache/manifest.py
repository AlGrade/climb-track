"""Versioned cache manifest models."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class OutputArtifact(BaseModel):
    """One materialized file belonging to a cache entry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    size_bytes: int
    sha256: str


class CacheManifest(BaseModel):
    """Reproducibility and integrity record for a stage result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0.0"
    status: Literal["incomplete", "complete"]
    stage: str
    stage_version: str
    cache_key: str
    created_at: datetime
    completed_at: datetime | None = None
    config_hash: str
    effective_config: dict[str, Any]
    input_fingerprint: dict[str, Any]
    tools: dict[str, Any]
    runtime: dict[str, Any]
    git: dict[str, Any]
    outputs: tuple[OutputArtifact, ...] = ()
