"""Stable fingerprints for immutable upstream cache entries."""

from climbtrack.cache.manifest import CacheManifest
from climbtrack.hashing import hash_json


def upstream_fingerprint(manifest: CacheManifest) -> dict[str, object]:
    """Summarize an upstream entry without depending on its filesystem path."""
    outputs = [output.model_dump(mode="json") for output in manifest.outputs]
    return {
        "stage": manifest.stage,
        "stage_version": manifest.stage_version,
        "cache_key": manifest.cache_key,
        "outputs_hash": hash_json(outputs),
    }
