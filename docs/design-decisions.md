# Design decisions

This file collects the reasoning behind choices that are not obvious from the code, especially the
ones where an easier option was rejected on purpose.

## What counts as a move

> A move is a hand move from one stable hand position to another stable hand position.

It starts when a previously quiet hand leaves its position and ends when the same hand stays quiet
at a new position for a minimum duration. Moves are classified as `left`, `right`, or `both`; when
both hands move inside a short shared window, that is one two-armed move, such as a dyno.

The definition needs several safety rules:

- A short correction on the same hold must not count as a new move.
- Body motion can begin before the hand and continue after the grab, so the player shows a small
  lead-in and lead-out around the boundary.
- Occlusion and wrong hand points can corrupt automatic boundaries. Every detection carries a
  confidence and stays correctable in the player.
- Without hold detection, the system can only infer contact from a quiet hand. It does not claim to
  have recognised the actual hold.
- "Moving hand" and "pulling hand" are not the same thing. The moving hand reaches for the next
  point while a quiet support hand may be doing the pulling. Forces are not measurable from a single
  2D video.

A completed move does not end at the new hand contact but once body and legs have settled too, or
when the next move begins. A terminal failed attempt is special-cased: after a fall there is by
definition no new stable hand position, so the detector combines the hand release with a clear
downward trunk motion and marks the move as `fall`, running from preparation to the lowest point.

### Why a palm point instead of a fingertip

Hand position is a robust palm point built from the wrist and several available hand points rather
than one fingertip, so the natural tremor of a single finger cannot fabricate a move.

The known weakness of that approach: the palm point jumps when the number of visible hand anchors
changes, because the median is then formed over a different set of points. On the reference video
this affects 21 of 1,648 frames, where speed runs at roughly one to two and a half times the
surrounding level. A minimum anchor count or a confidence-weighted palm would be the real fix.

## Why body length, not centimetres

Speeds are reported in `px/s` and in `body_lengths/s`. Real `cm/s` would be misleading without
calibration — that needs at least one known distance in the wall plane and ideally a static camera,
and perspective depth remains a limitation even then.

Body length is estimated **per frame** as an anatomical image chain from shoulder centre through hip
centre, knee, and ankle, smoothed over about a second. The climber does not need to stand upright,
because the chain is summed segment by segment and a bent knee does not shorten it in the image
plane. Depth changes slowly while per-frame foreshortening is noisy, which is why the value is
smoothed rather than taken raw — and why a single median over the whole video would be too coarse:
on the reference video the apparent body length varies between 295 and 446 pixels, which would have
mis-normalised individual moves by more than ten percent. `BL` is a relative comparison length, not
a measured real body size.

## Why the derivative window is small

Speed amplifies small position errors, so raw frame differences would be wrong. The denoising is
done by the **position** filter (a 9-frame median), not by the derivative window.

A comparison on the reference video makes this concrete: between radius 1 and radius 10 the noise
floor during quiet phases stays almost unchanged, while the captured peak speed drops from 100 to
52 percent. `speed_window_radius: 2` (about 84 ms) therefore keeps the peaks without buying noise.
The earlier 251 ms window blurred short moves badly.

All path columns use the same definition — the sum of actual frame steps — so `mean_speed_px_s` is
exactly `path_length_px / duration_seconds`.

## Why coordination has no threshold

`coordination_lag_seconds` answers whether the trunk starts before the hand. It deliberately uses
**no** threshold: a climber is rarely still before a move. On the reference video the trunk moves at
0.2 to 1.2 BL/s in the second before two of three moves, because feet are being set and weight
shifted. A threshold crossing would mostly measure the threshold.

Instead the two speed curves are cross-correlated; a positive value means the trunk went first.
`coordination_correlation` sits next to it so a weakly supported offset stays visible. If the
correlation maximum lands at the edge of the searched range, or one curve is flat, both fields stay
`null` rather than reporting an edge value as a measurement.

## Why refinement is selective

Stage 40 runs on the cached raw pose; Sapiens is not re-run. In order:

1. Confidence gating marks unreliable points explicitly as missing.
2. Implausible segment lengths reveal outliers on arms and legs.
3. Temporal consistency decides which endpoint of an implausible segment is the suspect.
4. Gaps of up to five frames are interpolated; longer gaps stay missing.
5. Sudden left/right swaps of symmetric points can be repaired.
6. An adaptive One-Euro filter smooths only the detailed hands by default.

Body and feet are deliberately not smoothed wholesale, so that dynos, swings, and falls remain the
genuinely fast movements they are. Raw and refined data stay side by side in the cache so any
improvement can be checked against the untouched model output at any time.

## Why the player tolerates failure

`climbtrack player` opens even when automatic move detection or the metrics stage rejects a video.
Both are only proposals to the player, and the player is the only place where boundaries can be
corrected. A hard abort would lock the reviewer out of the exact tool needed to fix the problem —
for instance after all moves were deleted, or when a new video yields no candidate at all. The
terminal states the reason, the move list stays empty, and the speed chart hides itself until
boundaries are set by hand and the player restarts.

The standalone `detect-moves` and `measure-moves` commands still fail loudly, because there the
result itself is the goal.

## Why the editor freezes the selection

While `Edit boundaries` is open, the video position no longer changes which move is selected.
Scrubbing to a new end frame necessarily leaves the current move, and the timeline-following
selection used to clear the draft when that happened: start frame and moving hand fell back to
unset, Save greyed out, and no message explained why. Corrections were impossible to save for
exactly the frames that needed them. An open editor now means edit intent, which outranks
timeline following. Collapsed, the selection follows the current frame as before.

## Reproducibility and safety choices

- Seeds are set as far as the backends allow deterministic behaviour.
- Configs reject unknown keys.
- Model and result files are verified by SHA-256.
- Cache entries record config, tool and package versions, operating system, and git provenance.
- Source timestamps and VFR are preserved; missing or non-monotonic timestamps are errors.
- ffmpeg decodes deterministically single-threaded and honours rotation metadata.
- There are no silent model, device, or HDR fallbacks.
- Raw and refined data stay separate so improvements remain verifiable.

## Known limitations

- Sapiens2-1B with four TTA passes per frame is very slow on Apple Silicon.
- No post-processing reliably reconstructs extreme occlusion.
- Long missing stretches stay missing on purpose.
- Face keypoints are stored but not rendered or annotated by default.
- The ground-truth set is one video and ten stress frames.
- Refinement clearly improves the right hand but not, on average, the left.
- The originally planned objective comparison against ViTPose is deliberately skipped.
- Perspective, wide angle, and camera motion are not converted into world coordinates.
- The palm point can jump when the number of visible hand anchors changes (see above).
