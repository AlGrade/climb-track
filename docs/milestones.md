# Development log

The project was built in two phases. Phase 1 answered "where is every body point in every frame";
Phase 2 turned that into moves and per-move measurements. This log records what each milestone
added and why it mattered for the end result.

## Phase 1 — accurate, temporally stable 2D skeletons

### Milestone 1 — foundation, ingest, and cache

- Python 3.12 project with `uv`, `pyproject.toml`, and a committed lock file.
- Strict YAML configuration: unknown options are rejected so typos cannot silently produce wrong
  results.
- Video analysed with ffprobe: resolution, rotation, HDR, variable frame rate, true timestamps.
- Every source frame extracted as PNG — PNG avoids adding JPEG artefacts in front of the models.
- Canonical frame parquet plus provenance for config, tools, Python, OS, and git.
- Content-addressed cache with SHA-256, atomic publish, manifests, and checksum verification.
- Failed builds are kept separately and never used as a valid cache entry.

**Why it matters:** every later measurement can be traced to one frame and its real timestamp, and
hours of work never have to be repeated.

### Milestone 2 — detection, tracking, and climber selection

- YOLO11x detects all people per frame, not just the most prominent one.
- ByteTrack links boxes over time and assigns stable track IDs.
- Overlapping partial-person boxes are suppressed confidence-aware.
- A transparent scoring picks the likely climber.
- Manual override via `--track-id` or `--click`; a hard stop when the choice is uncertain.
- Review video with box, confidence, track ID, frame number, and original audio.
- Pose crop brought to the official Sapiens aspect ratio of 768×1024 (3:4).
- Crop stabilised with motion context, 1.35× padding, short gap interpolation, and 15-frame
  smoothing. The crop was left deliberately generous so hands and feet do not fall out of the model
  image during wide moves.

**Why it matters:** other people do not confuse the pose estimator, and outstretched limbs stay
inside the Sapiens input.

### Milestone 3 — raw Sapiens2 pose

- Pinned Meta model `facebook/sapiens2-pose-1b` installed locally and verified by SHA-256.
- Official Transformers/safetensors implementation instead of the problematic old MMCV toolchain.
- Sapiens runs locally through Torch/MPS at 1024×768 input resolution.
- All 308 Goliath/Sociopticon keypoints stored per frame with unmodified confidence.
- Horizontal-flip TTA swaps left/right indices correctly; multi-scale TTA averages four forward
  passes per frame (`1.0` and `1.125`, each normal and mirrored).
- Raw pose written as `pose_raw.parquet`; no temporal correction yet.
- Frame-exact resume for the multi-hour model inference.
- Skeleton overlay rendered VFR-safe. An early bug where variable frame rate could drop frames was
  fixed; the reference video again contains 1,648 of 1,648 frames in the output.

Why not all 308 points appear in the video: 238 belong to the very dense face and are not drawn by
default so the overlay stays readable. Of the remaining 70 body, hand, and foot points, only those
above the drawing confidence threshold appear. The parquet file still holds all 308 values per
frame.

**Why it matters:** high-resolution raw measurements exist, so every later correction can be
compared objectively against untouched model output.

### Milestone 4 — ground truth and evaluation

- A lightweight local matplotlib annotation tool.
- Ten stress frames chosen deterministically: mostly low confidence and strong motion, plus timeline
  coverage.
- 40 movement-relevant points reviewed per frame: 17 body, 6 foot, 7 additional
  shoulder/elbow/neck points, and 10 fingertips.
- Points can be dragged or marked invisible. Every change is saved immediately and an interrupted
  session resumes later.
- Evaluation by pixel error, normalised PCK@0.2, and an OKS-like score, overall and split by body,
  extra points, feet, and hands.

```bash
uv run climbtrack annotate "/path/to/video.mp4" -c configs/default.yaml
uv run climbtrack evaluate annotations/<video-session>/ground_truth.json -c configs/default.yaml
```

In the editor: drag a wrong point, right-click an invisible one, then confirm the whole frame.

**Why it matters:** filters are tuned against real errors instead of gut feeling.

### Milestone 5 — conservative temporal refinement

Stage 40 uses only the existing `pose_raw.parquet`. The pipeline order and the reasoning behind
selective smoothing are described in [design-decisions.md](design-decisions.md#why-refinement-is-selective).

```bash
uv run climbtrack evaluate-refined "/path/to/video.mp4" \
  annotations/<video-session>/ground_truth.json -c configs/default.yaml
uv run climbtrack render-comparison "/path/to/video.mp4" -c configs/default.yaml
```

`raw_vs_refined.mp4` shows raw on the left and refined on the right.

**Why it matters:** the time series becomes reliable enough for derivatives without artificially
slowing down genuinely fast movement.

### Milestone 6 — skipped

A ViTPose comparison backend was planned and deliberately dropped. Phase 2 uses the existing refined
Sapiens data as its fixed source.

## Phase 2 — moves and movement metrics

Phase 2 builds directly on `pose_refined.parquet`; Sapiens does not run over the video again.

```text
pose_refined.parquet
        │
        ▼
hand and body time series
        │
        ▼
automatic move candidates ──► moves.parquet
        │                          │
        │                          ├──► local move player + manual correction
        │                          │
        │                          └──► per-move speed
        │
        └─────────────────────────────► per-move joint angles
                                            │
                                            ▼
                                     move_metrics.parquet
```

Original video timestamps remain the time base throughout. Speeds are never computed from an assumed
constant frame rate.

### P2.1 — move definition, data format, and player (implemented)

A local player shows the video with the skeleton drawn on it and offers two modes: play exactly one
move and pause at its end, or play the whole video with the active move marker and speed curve
following the current frame.

The interface offers **Previous move**, **Play move**, **Next move**, and **Full video**, and is
deliberately reduced to video, move list, and measurements. Frame stepping and the correction of
start, end, and hand assignment live under a collapsed **Edit boundaries**; changes are stored
separately as ground truth.

A boundary can be set two ways: `Use current frame` takes the current video position, or the frame
number is typed into the input field. `Enter` also seeks to that frame so it can be checked before
saving. Values outside the video are rejected rather than silently dropped, and an incomplete draft
names the missing fields under **Save** instead of leaving the button greyed out.

At startup the player loads `pose_refined.parquet`, forms a robust palm point per hand, and detects
transitions between stable positions. Existing pose and refinement caches are reused. For visible
playback it uses the cached `skeleton_raw_overlay.mp4`; boundaries themselves are computed from the
temporally smoothed data. Stage `90_player_video` derives a browser-friendly 1080-pixel version with
short keyframe intervals on first start, which makes seeking noticeably faster in Chrome.

Two desktop layouts are available. `Landscape layout` stacks video and chart compactly;
`Portrait layout` gives a 9:16 video a fitting window without wide black side panels. `Fullscreen`
switches only the video window, because in fullscreen only the skeleton should be judged; the
control bar there also shows `−1` and `+1` frame buttons. During playback the chrome fades after
about two seconds of quiet and returns on any input; while **paused** it stays, because that is when
frames are inspected one by one.

Every change is saved immediately and atomically to `annotations/<video-session>/`. A revision
number prevents two open browser tabs from overwriting each other unnoticed.

### P2.2 — evaluating and tuning automatic detection (reference video complete)

A conservative first detection combines hand speed from true source timestamps, stable phases before
and after the movement, the settling of body and legs after hand contact, minimum displacement and
duration, a robust multi-point palm position, and body-size-normalised thresholds. A terminal failed
attempt is detected separately. See
[design-decisions.md](design-decisions.md#what-counts-as-a-move) for the full reasoning.

Count, hand side, and boundaries were reviewed in the player on the reference video. That feedback
set the start threshold, the end after body settling, and the special handling of a terminal fall.
Manual review remains the safety net for further videos; two-handed overlap and explicit uncertainty
warnings are possible later extensions.

### P2.3 — per-move speed (implemented)

For the **moving hand**, per move: duration; horizontal, vertical, and total path; direct
displacement and actual path length; mean and maximum speed; and the timestamp of maximum speed.

For the **body**, a robust trunk centre from shoulders and hips gives body path, vertical
displacement, and mean and maximum speed. The movement of the quiet hand relative to the body can
also be described — a hint at a support phase, not a force measurement.

Two additional groups target the comparison of **two attempts at the same move**, both chosen to be
insensitive to camera position: `hand_settle_*` marks the frame where the moving hand comes to rest
(the grab, or the lowest point of a fall), and on top of it `hip_rise_body_lengths` and
`hip_below_hand_body_lengths` describe how far the hip rose and how extended the body was at the
grab. `coordination_lag_seconds` and `coordination_correlation` describe trunk lead.

Units, normalisation, and the derivative window are explained in
[design-decisions.md](design-decisions.md#why-body-length-not-centimetres).

### P2.4 — per-move joint angles (planned)

First sensible 2D angles: left and right elbow, left and right shoulder in the image, left and right
knee, left and right hip, and trunk inclination. Per move, angles at start and end plus minimum,
maximum, and range of motion. Frames with missing or too uncertain joints are marked invalid rather
than invented.

All values are **2D image angles**. When the climber turns towards the wall or out of the image
plane, they are not identical to true anatomical 3D joint angles.

### P2.5 — extended result view and export (planned)

The first speed card already ships in P2.3. P2.5 extends the player with detailed paths and height
gain, selected angles and range of motion, and warnings for missing or uncertain data. Machine
readable results stay in Parquet/JSON; a compact CSV export is planned so individual moves can be
compared later.
