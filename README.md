# ClimbTrack

ClimbTrack is an offline, quality-first pipeline for temporally stable 2D skeleton tracking in
bouldering and climbing videos. Phase 1 deliberately excludes speed, movement metrics, 3D,
hold detection, a server, and an end-user UI.

The project is currently for private use only. Sapiens2-1B is the approved primary pose backend,
subject to its license terms. Milestone 2 uses Ultralytics YOLO11x/ByteTrack; review the
Ultralytics AGPL-3.0 terms before any use beyond this private project.

## Current status

Milestones 1 and 2 implement:

- strict YAML configuration with unknown-key rejection;
- Stage 00 video probing and lossless PNG extraction;
- source timestamps, VFR, rotation and HDR detection;
- a content-addressed, atomic cache with checksum verification and resume;
- provenance for inputs, tools, Python, OS and Git;
- canonical frame and pose Parquet schemas with explicit missing keypoints;
- pinned YOLO11x person detection at configurable inference resolution;
- ByteTrack association with canonical detection and track Parquet schemas, including
  confidence-aware containment suppression for duplicate partial-person boxes;
- explainable climber ranking from track length, continuity, movement, position and area;
- explicit `--track-id` and interactive `--click` selection;
- a hard uncertainty stop when automatic selection is ambiguous;
- a VFR-aware MP4 overlay with bounding boxes, confidence, track IDs, frame numbers and audio;
- resumable CLI commands for each stage and `run-all`;
- unit tests for hashing, cache behavior, timestamps, schemas, scoring, ByteTrack and VFR output.

Pose inference, skeleton rendering, refinement, annotation and evaluation remain outside this
milestone. The current overlay verifies person detection, identity persistence and climber choice.

## Requirements

- macOS on Apple Silicon (other platforms are supported where the configured tools exist)
- `uv`
- Python 3.12, managed by `uv`
- `ffmpeg` and `ffprobe`
- enough disk space for lossless PNG ingest plus temporary overlay JPEGs

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

No unavailable device or model is replaced automatically. A configured `mps`, `cpu`, or `cuda`
device that cannot execute the selected backend fails clearly. Change the YAML explicitly if you
intend to use CPU or a rented CUDA host.

## Usage

Download the pinned YOLO11x checkpoint explicitly (there are no implicit downloads):

```bash
uv run climbtrack download-yolo --config configs/default.yaml
```

Validate configuration, external tools, checkpoint and configured device:

```bash
uv run climbtrack preflight --config configs/default.yaml
```

Run Stage 00:

```bash
uv run climbtrack ingest /absolute/path/to/video.mov --config configs/default.yaml
```

Run individual Milestone-2 stages (prerequisites are resumed from cache):

```bash
uv run climbtrack detect /absolute/path/to/video.mov
uv run climbtrack track /absolute/path/to/video.mov
uv run climbtrack select /absolute/path/to/video.mov
uv run climbtrack render-tracks /absolute/path/to/video.mov
```

Run every currently implemented stage:

```bash
uv run climbtrack run-all /absolute/path/to/video.mov --config configs/default.yaml
```

If automatic climber selection reports uncertainty, inspect the ranked IDs and confirm one:

```bash
uv run climbtrack run-all /absolute/path/to/video.mov --track-id 7
```

To create an overlay containing every candidate ID before deciding:

```bash
uv run climbtrack render-tracks /absolute/path/to/video.mov --review-all
```

On a local desktop session you can instead select a displayed box:

```bash
uv run climbtrack run-all /absolute/path/to/video.mov --click
```

Inspect all complete cache entries:

```bash
uv run climbtrack cache-list --config configs/default.yaml
```

`--force` rebuilds the same cache key but moves the previous entry to a recoverable hidden backup;
it does not delete prior results.

## Cache contract

Entries are stored by deterministic stage-specific cache key:

```text
cache/00_ingest/<sha256-cache-key>/
├── manifest.json
├── metadata.json
├── ffprobe.json
├── frames.parquet
└── frames/
    ├── 000000000.png
    └── ...
cache/10_detect/<cache-key>/detections.parquet
cache/20_track/<cache-key>/tracks.parquet
cache/25_select/<cache-key>/{candidates.json,selection.json}
cache/50_render_tracks/<cache-key>/tracking_overlay.mp4
```

The cache key covers the source video's full SHA-256, the effective Stage-00 configuration, the
stage implementation version, and the resolved ffmpeg/ffprobe versions. A complete manifest stores
checksums for every artifact. Builds are published by atomic directory rename; interrupted builds
are retained under `.failed/` and are never treated as cache hits.

Only a stage's effective configuration participates in its cache key. A rendering change therefore
cannot invalidate tracking or ingest artifacts. Downstream keys include the upstream artifact hash;
detection also includes the exact checkpoint SHA-256 and package versions.

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

## Model download policy

YOLO11x is stored at `models/yolo11x.pt`, outside Git. The explicit `download-yolo` command uses
the pinned URL from YAML, writes to a temporary file, publishes atomically and fingerprints the
result. Inference never triggers a download.

For Milestone 3, pose models will likewise remain outside Git. The approved primary model is:

Models are stored outside Git and are never downloaded implicitly. The approved primary model is:

```text
repository: facebook/sapiens2-pose-1b
file:       sapiens2_1b_pose.safetensors
```

The exact Hugging Face revision and downloaded SHA-256 must be pinned before inference. ViTPose++
Huge will use the maintained Transformers/Safetensors path rather than legacy MMCV 1.3.9.

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
