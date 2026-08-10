# ClimbTrack

ClimbTrack is an offline, quality-first pipeline for temporally stable 2D skeleton tracking in
bouldering and climbing videos. Phase 1 deliberately excludes speed, movement metrics, 3D,
hold detection, a server, and an end-user UI.

The project is currently for private use only. Sapiens2-1B is the approved primary pose backend,
subject to its license terms. Milestone 1 does not download or execute any model.

## Current status

Milestone 1 implements:

- strict YAML configuration with unknown-key rejection;
- Stage 00 video probing and lossless PNG extraction;
- source timestamps, VFR, rotation and HDR detection;
- a content-addressed, atomic cache with checksum verification and resume;
- provenance for inputs, tools, Python, OS and Git;
- canonical frame and pose Parquet schemas with explicit missing keypoints;
- CLI commands `preflight`, `ingest`, `run-all`, and `cache-list`;
- unit tests for hashing, cache behavior, timestamps, rotation, VFR and schema invariants.

Detection, tracking, person selection, pose inference, refinement, rendering, annotation and
evaluation intentionally remain outside this milestone.

## Requirements

- macOS on Apple Silicon (other platforms are supported where the configured tools exist)
- `uv`
- Python 3.12, managed by `uv`
- `ffmpeg` and `ffprobe`

Install ffmpeg with Homebrew if `climbtrack preflight` reports that it is missing:

```bash
brew install ffmpeg
```

Create the environment and install the locked project dependencies:

```bash
uv sync --locked
```

If no lockfile exists yet, use `uv sync` once and commit the resulting `uv.lock`.

## Configuration

Copy `configs/default.yaml` for experiments instead of editing model or processing parameters in
code. Relative cache paths are resolved against the project root containing `configs/`.

Important ingest settings:

- `ffmpeg_path` and `ffprobe_path` may be executable names in `PATH` or explicit paths.
- `hdr_policy: fail` is the quality-safe default.
- `hdr_policy: tonemap` performs an explicit Hable tone map through ffmpeg's `zscale`/`tonemap`
  filters. The local ffmpeg build must provide those filters.
- `hdr_policy: clip` explicitly accepts direct conversion to 8-bit RGB and can lose highlight
  detail. It must never be selected implicitly.
- `verify_cached_checksums: true` verifies every artifact before declaring a cache hit.

No unavailable device or model is replaced automatically. Later model stages will fail clearly if
the configured `mps`, `cpu`, or `cuda` device cannot execute the selected backend.

## Usage

Validate configuration and external tools:

```bash
uv run climbtrack preflight --config configs/default.yaml
```

Run Stage 00:

```bash
uv run climbtrack ingest /absolute/path/to/video.mov --config configs/default.yaml
```

Run every currently implemented stage:

```bash
uv run climbtrack run-all /absolute/path/to/video.mov --config configs/default.yaml
```

Inspect complete ingest cache entries:

```bash
uv run climbtrack cache-list --config configs/default.yaml
```

`--force` rebuilds the same cache key but moves the previous entry to a recoverable hidden backup;
it does not delete prior results.

## Cache contract

Stage 00 entries are stored under:

```text
cache/00_ingest/<sha256-cache-key>/
├── manifest.json
├── metadata.json
├── ffprobe.json
├── frames.parquet
└── frames/
    ├── 000000000.png
    └── ...
```

The cache key covers the source video's full SHA-256, the effective Stage-00 configuration, the
stage implementation version, and the resolved ffmpeg/ffprobe versions. A complete manifest stores
checksums for every artifact. Builds are published by atomic directory rename; interrupted builds
are retained under `.failed/` and are never treated as cache hits.

Only a stage's effective configuration participates in its cache key. A future rendering change
therefore cannot invalidate pose or ingest artifacts. Downstream stages will add their upstream
artifact hash and exact model/checkpoint identity.

## Timestamp and image policy

ClimbTrack preserves `best_effort_timestamp_time` from ffprobe for each decoded source frame.
It does not reconstruct timestamps from an assumed FPS. Missing or non-monotonic timestamps are a
hard error. ffmpeg's default autorotation is applied during frame extraction; source rotation and
both encoded and display dimensions are retained in `metadata.json`.

PNG is used to avoid adding JPEG artifacts before person detection and pose estimation. ffmpeg is
run single-threaded for deterministic output and its exact version is part of the cache key.

## Canonical pose schema

The long-form Parquet schema is versioned and contains:

```text
frame_idx, timestamp, track_id, keypoint_name,
x, y, confidence, is_missing, is_interpolated, source_backend
```

A missing point must use Arrow nulls for `x`, `y`, and `confidence`; zero coordinates are invalid.
The future 308-keypoint registry will additionally version backend mappings, symmetry partners,
body groups and skeleton edges from the official model metadata.

## Model download policy for Milestone 3

Models are stored outside Git and are never downloaded implicitly. The approved primary model is:

```text
repository: facebook/sapiens2-pose-1b
file:       sapiens2_1b_pose.safetensors
```

The exact Hugging Face revision and downloaded SHA-256 must be pinned before inference. A typical
manual download will use `hf download` only after the Sapiens2 runtime and license checkpoint in
Milestone 3. ViTPose++ Huge will use the maintained Transformers/Safetensors path rather than the
legacy MMCV 1.3.9 installation.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

The integration test creates a tiny synthetic video and is skipped when ffmpeg or ffprobe is not
installed. Tests cover pure processing logic, not neural-network inference.

## Planned stages

```text
00_ingest  -> frames and source metadata
10_detect  -> YOLO11 person detections
20_track   -> ByteTrack tracklets
25_select  -> explicit climber selection with uncertainty stop
30_pose    -> Sapiens2 / ViTPose raw observations
40_refine  -> gating, swap/outlier handling, interpolation, One-Euro filtering
50_render  -> visual quality-control overlays
60_eval    -> OKS, PCK and temporal/backend comparisons
```

