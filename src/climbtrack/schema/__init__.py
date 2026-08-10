"""Canonical, backend-independent data schemas."""

from climbtrack.schema.pose import POSE_SCHEMA, write_pose_parquet

__all__ = ["POSE_SCHEMA", "write_pose_parquet"]
