/** Lookups between video time, timeline position, and source frame index.
 *
 * The source video has a variable frame rate, so nothing here may derive a time
 * from an assumed constant fps. Every answer comes from the frame index the
 * server sent, which carries the real source timestamps.
 */

import { elements } from "./dom.js";
import { state } from "./state.js";

/** Binary search for the timeline entry closest to a media time. */
export function nearestTimelineIndex(time) {
  let low = 0;
  let high = state.timeline.length - 1;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (state.timeline[middle].media_time < time) low = middle + 1;
    else high = middle;
  }
  if (low > 0) {
    const before = state.timeline[low - 1];
    const after = state.timeline[low];
    if (Math.abs(before.media_time - time) <= Math.abs(after.media_time - time)) return low - 1;
  }
  return low;
}

export function nearestFrame(time = elements.video.currentTime) {
  return state.timeline[nearestTimelineIndex(time)];
}

/** Media time of a source frame index, or 0 when the frame is unknown. */
export function mediaTime(frameIdx) {
  const timelineIndex = state.timelineIndexByFrame.get(frameIdx);
  const frame = timelineIndex === undefined ? null : state.timeline[timelineIndex];
  return frame ? frame.media_time : 0;
}

export function hasFrame(frameIdx) {
  return state.timelineIndexByFrame.has(frameIdx);
}

export function timelineIndexOf(frameIdx) {
  return state.timelineIndexByFrame.get(frameIdx);
}

export function lastFrameIdx() {
  if (!state.timeline.length) return 0;
  return state.timeline[state.timeline.length - 1].frame_idx;
}

/** Store the timeline and the frame-index lookup built from it. */
export function loadTimeline(timeline) {
  state.timeline = timeline;
  state.timelineIndexByFrame = new Map(timeline.map((frame, index) => [frame.frame_idx, index]));
}
