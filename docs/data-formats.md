# Data formats and cache layout

Every expensive step writes its result into a content-addressed cache. Directory names are a
deterministic hash of the video, the relevant configuration, the implementation, the model
identities, and the direct upstream artefacts. Change any of those and you get a new entry rather
than a silently stale one.

## Cache layout

```text
cache/
├── 00_ingest/<key>/
│   ├── frames/000000000.png    one lossless PNG per source frame
│   ├── frames.parquet          frame index with true source timestamps
│   ├── metadata.json           config, tool versions, git and runtime provenance
│   └── ffprobe.json            raw probe output
├── 10_detect/<key>/detections.parquet
├── 20_track/<key>/tracks.parquet
├── 25_select/<key>/
│   ├── candidates.json         every scored track, so a decision can be reviewed
│   ├── selection.json          the chosen climber and how it was chosen
│   └── pose_crops.parquet      stabilised 3:4 crop per frame
├── 30_pose/<key>/
│   ├── pose_raw.parquet        308 keypoints per frame, unmodified model output
│   ├── keypoints.json          versioned registry: names, groups, skeleton edges
│   └── summary.json
├── 40_refine/<key>/
│   ├── pose_refined.parquet    gated, repaired and selectively smoothed
│   └── summary.json
├── 70_moves/<key>/
│   ├── moves_auto.parquet      automatic move candidates
│   └── summary.json
├── 80_move_metrics/<key>/
│   ├── move_metrics.parquet    one row per move
│   ├── move_speed_timeline.parquet
│   ├── move_metrics.json
│   └── summary.json
├── 50_render_tracks/<key>/tracking_overlay.mp4
├── 50_render_pose/<key>/skeleton_raw_overlay.mp4
├── 50_render_compare/<key>/raw_vs_refined.mp4
└── 90_player_video/<key>/player_video.mp4

annotations/<video-session>/
├── ground_truth.json           hand-corrected keypoints (versioned)
├── moves_ground_truth.json     hand-corrected move boundaries (versioned)
├── moves.parquet               derived from the JSON on every save (ignored by git)
├── evaluation.json             derived (ignored by git)
└── evaluation_refined.json     derived (ignored by git)
```

### Size

The lossless input frames dominate everything else: roughly **0.4 GB per second of 4K footage**.
A 27-second reference video occupies about 12 GB of frames out of a 13 GB cache. Model weights add
another 5.8 GB. Neither `cache/` nor `models/` belongs in version control, and both are ignored.

### Invalidation and safety

If only the rendering configuration changes, ingest, detection, tracking, and pose stay valid. If a
pose setting or an upstream artefact changes, a new cache key is created on purpose. Complete
entries carry a manifest with checksums for every artefact. A build is published by atomically
renaming its directory, so a crash can never leave a half-written entry that later looks valid.

## Pose parquet

Pose data is stored one row per keypoint per frame:

```text
frame_idx, timestamp, track_id, keypoint_name,
x, y, confidence, is_missing, is_interpolated, source_backend
```

Rules that the rest of the pipeline relies on:

- Coordinates are in the **original image**, not in the crop.
- `timestamp` comes from the source video. It is never reconstructed from an assumed frame rate,
  because the input is variable-frame-rate and that reconstruction would drift.
- Missing points use real Arrow nulls for `x`, `y`, and `confidence` — never `(0, 0)`, which would
  silently become a plausible-looking measurement downstream.
- Raw confidence is preserved even where the value is low.
- The registry, group assignment, left/right pairs, and skeleton edges are versioned alongside.
- A different pose backend can later be mapped into the same canonical schema.

## Move schema

```text
move_id, start_frame, end_frame, start_timestamp, end_timestamp,
moving_hand, confidence, source, is_reviewed, outcome
```

`moving_hand` is `left`, `right`, or `both`. `source` is `automatic`, `manual`, or `corrected`.
`outcome` separates `completed` from `fall`.

`moves_ground_truth.json` is the editable session and the single source of truth; `moves.parquet`
holds the same state for the metrics stage and is rewritten on every save. A revision number
prevents two open browser tabs from overwriting each other unnoticed.

## Configuration

All parameters live in `configs/default.yaml`. Unknown keys are rejected rather than ignored, so a
typo cannot quietly produce different results. For experiments, copy the file and pass your own via
`--config` instead of editing values in Python.

| Section | What it controls |
|---|---|
| `project` | cache and annotation directories, seed, compute device (`mps`, `cpu`, `cuda`) |
| `ingest` | ffmpeg/ffprobe paths, PNG settings, HDR policy, cache verification |
| `detection` | YOLO input resolution and thresholds |
| `tracking` | ByteTrack matching and track buffer |
| `selection` | minimum quality and the weighted climber score |
| `pose_crop` | aspect ratio, padding, temporal crop stabilisation |
| `pose` | batch size, precision, flip and multi-scale TTA |
| `pose_render` | visibility of face, crop, points, comparison panel width |
| `annotation` | number of stress frames, PCK and OKS parameters |
| `refine` | confidence thresholds, gap size, One-Euro, segment and swap parameters |
| `move_player` | host, port, lead-in and lead-out around a move |
| `move_detection` | smoothing, stability and displacement thresholds for segmentation |
| `move_metrics` | derivative window, validity floor, settle and coordination parameters |
| `models` | immutable model revisions, file sizes, and SHA-256 hashes |

HDR input is rejected by default (`hdr_policy: fail`). `tonemap` requires matching ffmpeg filters;
`clip` deliberately accepts possible highlight loss. No lossy policy is ever chosen silently.

## Model pinning

YOLO11x lives at `models/yolo11x.pt`. The primary pose model is pinned by revision and checksum:

```text
Repository: facebook/sapiens2-pose-1b
Revision:   f5fed8b97b99698d5eea1d14ff0855d0b4c3f000
File:       model.safetensors
Size:       6.08 GB
```

The official keypoint metadata is pinned separately to Sapiens2 commit
`7e5bae88456ac418ff0e58e74106c9fe192055d4`. It is read as data; downloaded Python code is never
executed. Downloads land in a temporary file, are verified against size and SHA-256, and only then
published atomically. Inference runs `local-only` and does not contact Hugging Face.
