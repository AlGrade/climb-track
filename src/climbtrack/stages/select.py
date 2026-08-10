"""Stage 25: explainable climber selection with uncertainty stop."""

import json
from pathlib import Path

from climbtrack.cache import CacheResult, StageCache
from climbtrack.cache.upstream import upstream_fingerprint
from climbtrack.config import AppConfig
from climbtrack.errors import SelectionUncertainError, UnknownTrackError
from climbtrack.provenance import git_state, runtime_state
from climbtrack.schema.tracks import read_tracks
from climbtrack.selection.scoring import SelectionCandidate, rank_candidates

STAGE_NAME = "25_select"
STAGE_VERSION = "1.0.0"


def select_climber(
    ingest: CacheResult,
    tracks: CacheResult,
    *,
    config: AppConfig,
    cache_root: Path,
    project_root: Path,
    manual_track_id: int | None = None,
    force: bool = False,
) -> CacheResult:
    """Select the climber or fail with ranked candidates when confidence is insufficient."""
    rows = read_tracks(tracks.path / "tracks.parquet")
    metadata = json.loads((ingest.path / "metadata.json").read_text(encoding="utf-8"))
    candidates = rank_candidates(
        rows,
        image_width=int(metadata["video"]["display_width"]),
        image_height=int(metadata["video"]["display_height"]),
        config=config.selection,
    )
    selected, method, margin = _decide(candidates, config, manual_track_id)

    effective_config = {
        **config.selection.model_dump(mode="json"),
        "manual_track_id": manual_track_id,
    }
    tools: dict[str, object] = {}
    input_fingerprint = upstream_fingerprint(tracks.manifest)
    runtime = runtime_state()
    git = git_state(project_root)
    cache = StageCache(cache_root, STAGE_NAME)
    cache_key = cache.make_key(
        stage=STAGE_NAME,
        stage_version=STAGE_VERSION,
        effective_config=effective_config,
        input_fingerprint=input_fingerprint,
        tools=tools,
    )

    def build(output: Path) -> None:
        candidate_payload = [candidate.as_dict() for candidate in candidates]
        (output / "candidates.json").write_text(
            json.dumps(candidate_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        selection = {
            "schema_version": "1.0.0",
            "stage": STAGE_NAME,
            "track_id": selected.track_id,
            "method": method,
            "score": selected.score,
            "score_margin": margin,
            "candidate": selected.as_dict(),
            "effective_config": effective_config,
        }
        (output / "selection.json").write_text(
            json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    return cache.materialize(
        cache_key=cache_key,
        stage_version=STAGE_VERSION,
        effective_config=effective_config,
        input_fingerprint=input_fingerprint,
        tools=tools,
        runtime=runtime,
        git=git,
        verify_checksums=True,
        force=force,
        builder=build,
    )


def _decide(
    candidates: list[SelectionCandidate],
    config: AppConfig,
    manual_track_id: int | None,
) -> tuple[SelectionCandidate, str, float | None]:
    payload = [candidate.as_dict() for candidate in candidates]
    if manual_track_id is not None:
        selected = next(
            (candidate for candidate in candidates if candidate.track_id == manual_track_id), None
        )
        if selected is None:
            raise UnknownTrackError(f"Track ID {manual_track_id} does not exist")
        return selected, "manual_track_id", None

    eligible = [candidate for candidate in candidates if candidate.eligible]
    if not eligible:
        raise SelectionUncertainError(
            "No track meets the minimum observation and continuity requirements", payload
        )
    selected = eligible[0]
    margin = selected.score - eligible[1].score if len(eligible) > 1 else 1.0
    if margin < config.selection.minimum_score_margin:
        raise SelectionUncertainError(
            f"Top-candidate score margin {margin:.3f} is below "
            f"the required {config.selection.minimum_score_margin:.3f}",
            payload,
        )
    return selected, "automatic", margin
