/** Playing, seeking, scrubbing, and stepping the video frame by frame.
 *
 * Everything here works in timeline indices rather than seconds, because the
 * source video has a variable frame rate and "one frame" is not a fixed duration.
 */

import { updateChartCursor } from "./chart.js";
import { FRAME_PRESENT_TIMEOUT_MS, MAX_QUEUED_FRAME_STEPS } from "./constants.js";
import { elements } from "./dom.js";
import { formatControlTime, formatTime } from "./format.js";
import { updateNavigation } from "./move-list.js";
import { selectMove, syncMoveSelection } from "./selection.js";
import { state } from "./state.js";
import { mediaTime, nearestTimelineIndex, timelineIndexOf } from "./timeline.js";
import { setSaveState } from "./status.js";

// ---------------------------------------------------------------- playing

/** Play exactly one move, then pause at its end. */
export function playMove(index) {
  if (!state.session.moves.length) return;
  selectMove(index);
  const move = state.session.moves[state.selectedIndex];
  const start = Math.max(0, mediaTime(move.start_frame) - state.settings.lead_in_seconds);
  let end = Math.min(
    Number.isFinite(elements.video.duration) ? elements.video.duration : Infinity,
    mediaTime(move.end_frame) + state.settings.lead_out_seconds,
  );
  const nextMove = state.session.moves[state.selectedIndex + 1];
  if (nextMove) end = Math.min(end, mediaTime(nextMove.start_frame));
  startPlayback(start, end);
}

/** Play from `start`, pausing at `end` (null plays to the end of the video). */
export function startPlayback(start, end) {
  cancelFrameStepping();
  state.playbackReady = false;
  state.playbackEnd = end;

  // Keep play() in the original click task, then seek once Chrome has accepted
  // playback. Seeking first can make Chrome interrupt the play request.
  elements.video
    .play()
    .then(() => {
      const needsSeek =
        Math.abs(elements.video.currentTime - start) >= 0.01 || elements.video.seeking;
      if (!needsSeek) {
        state.playbackReady = true;
        return;
      }
      elements.video.addEventListener(
        "seeked",
        () => {
          state.playbackReady = true;
          window.requestAnimationFrame(() => {
            elements.video.play().catch(reportPlaybackError);
          });
        },
        { once: true },
      );
      elements.video.currentTime = start;
    })
    .catch(reportPlaybackError);
}

export function toggleVideoPlayback() {
  cancelFrameStepping();
  state.playbackEnd = null;
  if (!elements.video.paused) {
    elements.video.pause();
    return;
  }
  state.selectionFollowsTimeline = true;
  if (elements.video.ended) startPlayback(0, null);
  else elements.video.play().catch(reportPlaybackError);
}

function reportPlaybackError(error) {
  state.playbackEnd = null;
  state.playbackReady = true;
  elements.activeMove.textContent = `Playback failed: ${error.message}`;
  console.error("Video playback failed", error);
}

/** Drive the readout while playing and stop at the end of a move. */
export function watchPlayback() {
  if (state.watcherRunning) return;
  state.watcherRunning = true;
  const tick = () => {
    updateReadout();
    if (
      state.playbackReady
      && state.playbackEnd !== null
      && elements.video.currentTime >= state.playbackEnd
    ) {
      elements.video.pause();
      elements.video.currentTime = state.playbackEnd;
      state.playbackEnd = null;
      setSaveState("Move complete", "saved");
    }
    if (elements.video.paused) state.watcherRunning = false;
    else window.requestAnimationFrame(tick);
  };
  window.requestAnimationFrame(tick);
}

// ---------------------------------------------------------------- seeking

export function seekToFrame(frameIdx) {
  const timelineIndex = timelineIndexOf(frameIdx);
  if (timelineIndex === undefined) return;
  seekToTimelineIndex(timelineIndex);
}

export function seekToSelectedMoveStart() {
  const move = state.session.moves[state.selectedIndex];
  if (move) seekToFrame(move.start_frame);
}

function seekToTimelineIndex(timelineIndex, options = {}) {
  if (!options.frameStep) cancelFrameStepping();
  elements.video.pause();
  state.playbackEnd = null;
  state.playbackReady = true;
  elements.video.currentTime = state.timeline[timelineIndex].media_time;
  updateReadoutForTimelineIndex(timelineIndex);
}

// ---------------------------------------------------------- frame stepping

/**
 * Step exactly one frame.
 *
 * A held arrow key can outrun the decoder, so further steps are queued and
 * released only once the browser reports the previous frame as presented.
 */
export function stepFrame(offset) {
  if (!state.timeline.length) return;
  if (state.frameStepInFlight) {
    state.queuedFrameSteps = Math.max(
      -MAX_QUEUED_FRAME_STEPS,
      Math.min(MAX_QUEUED_FRAME_STEPS, state.queuedFrameSteps + offset),
    );
    return;
  }
  const current = nearestTimelineIndex(elements.video.currentTime);
  const target = Math.max(0, Math.min(current + offset, state.timeline.length - 1));
  if (target === current) return;
  state.frameStepInFlight = true;
  seekToTimelineIndex(target, { frameStep: true });
}

export function holdFrameDirection(direction) {
  state.heldFrameDirection = direction;
}

export function releaseFrameDirection(direction) {
  if (state.heldFrameDirection !== direction) return;
  state.heldFrameDirection = 0;
  if (state.frameHoldTimer !== null) {
    window.clearTimeout(state.frameHoldTimer);
    state.frameHoldTimer = null;
  }
}

function finishFrameStep() {
  if (!state.frameStepInFlight) return;
  state.frameStepInFlight = false;
  let nextDirection = state.heldFrameDirection;
  if (nextDirection === 0 && state.queuedFrameSteps !== 0) {
    nextDirection = Math.sign(state.queuedFrameSteps);
    state.queuedFrameSteps -= nextDirection;
  }
  if (nextDirection === 0) return;
  state.frameHoldTimer = window.setTimeout(() => {
    state.frameHoldTimer = null;
    stepFrame(nextDirection);
  }, 0);
}

function cancelFrameStepping() {
  state.frameStepInFlight = false;
  state.queuedFrameSteps = 0;
  state.heldFrameDirection = 0;
  if (state.frameHoldTimer !== null) {
    window.clearTimeout(state.frameHoldTimer);
    state.frameHoldTimer = null;
  }
}

// --------------------------------------------------------------- scrubbing

export function beginScrub() {
  if (state.scrubbing) return;
  cancelFrameStepping();
  elements.video.pause();
  state.playbackEnd = null;
  state.playbackReady = true;
  state.scrubbing = true;
  state.scrubTargetIndex = nearestTimelineIndex(elements.video.currentTime);
  state.scrubAppliedIndex = state.scrubTargetIndex;
  state.scrubFinishing = false;
}

/** Update the picture while dragging, at most once per animation frame. */
export function previewScrub() {
  if (!state.scrubbing) beginScrub();
  state.scrubTargetIndex = Number(elements.videoScrubber.value);
  updateReadoutForTimelineIndex(state.scrubTargetIndex);
  scheduleScrubSeek();
}

export function finishScrub() {
  if (!state.scrubbing) return;
  state.scrubTargetIndex ??= Number(elements.videoScrubber.value);
  state.scrubFinishing = true;
  scheduleScrubSeek();
  completeScrubIfReady();
}

function scheduleScrubSeek() {
  if (
    state.scrubAnimationFrame !== null
    || state.scrubSeekInFlight
    || state.scrubTargetIndex === null
  ) {
    return;
  }
  state.scrubAnimationFrame = window.requestAnimationFrame(() => {
    state.scrubAnimationFrame = null;
    if (state.scrubTargetIndex === null) return;
    const currentIndex = nearestTimelineIndex(elements.video.currentTime);
    if (state.scrubTargetIndex === currentIndex && !elements.video.seeking) {
      state.scrubAppliedIndex = state.scrubTargetIndex;
      completeScrubIfReady();
      return;
    }
    state.scrubSeekInFlight = true;
    state.scrubAppliedIndex = state.scrubTargetIndex;
    elements.video.currentTime = state.timeline[state.scrubTargetIndex].media_time;
  });
}

function completeScrubIfReady() {
  if (
    !state.scrubFinishing
    || state.scrubSeekInFlight
    || state.scrubAnimationFrame !== null
    || state.scrubTargetIndex !== state.scrubAppliedIndex
  ) {
    return;
  }
  const target = state.scrubTargetIndex;
  state.scrubbing = false;
  state.scrubTargetIndex = null;
  state.scrubAppliedIndex = null;
  state.scrubFinishing = false;
  updateReadoutForTimelineIndex(target);
}

// ----------------------------------------------------------------- readout

export function updateReadout() {
  if (!state.timeline.length) return;
  if (state.scrubbing && state.scrubTargetIndex !== null) {
    updateReadoutForTimelineIndex(state.scrubTargetIndex);
    return;
  }
  updateReadoutForTimelineIndex(nearestTimelineIndex(elements.video.currentTime));
}

function updateReadoutForTimelineIndex(timelineIndex) {
  const frame = state.timeline[timelineIndex];
  if (!frame) return;
  elements.currentTime.textContent = formatTime(frame.media_time);
  elements.currentFrame.textContent = String(frame.frame_idx);
  elements.videoScrubber.value = String(timelineIndex);
  const progress =
    state.timeline.length <= 1 ? 0 : (timelineIndex / (state.timeline.length - 1)) * 100;
  elements.videoScrubber.style.setProperty("--scrub-progress", `${progress.toFixed(3)}%`);
  const duration = Number.isFinite(elements.video.duration)
    ? elements.video.duration
    : state.timeline[state.timeline.length - 1].media_time;
  elements.videoTime.textContent =
    `${formatControlTime(frame.media_time)} / ${formatControlTime(duration)}`;
  syncMoveSelection(frame.frame_idx);
  updateChartCursor(frame.media_time);
}

export function updatePlaybackControls() {
  const playing = !elements.video.paused && !elements.video.ended;
  elements.videoToggle.textContent = playing ? "❚❚" : "▶";
  elements.videoToggle.setAttribute("aria-label", playing ? "Pause video" : "Play video");
  elements.videoMute.textContent = elements.video.muted ? "🔇" : "🔊";
  elements.videoMute.setAttribute(
    "aria-label",
    elements.video.muted ? "Unmute video" : "Mute video",
  );
}

/** Called once the video reports its duration. */
export function configureVideoControls() {
  elements.videoScrubber.max = String(Math.max(0, state.timeline.length - 1));
  updateNavigation();
  updatePlaybackControls();
  updateReadout();
}

export function handleVideoSeeked() {
  updateReadout();
  if (state.scrubbing) {
    afterVideoFramePresented(() => {
      if (!state.scrubbing) return;
      state.scrubSeekInFlight = false;
      if (state.scrubTargetIndex !== state.scrubAppliedIndex) scheduleScrubSeek();
      else completeScrubIfReady();
    });
    return;
  }
  if (state.frameStepInFlight) afterVideoFramePresented(finishFrameStep);
}

/**
 * Run a callback once the seeked-to frame is actually on screen.
 *
 * `requestVideoFrameCallback` is the accurate signal but is not universal, and it
 * never fires for a frame the browser decides not to repaint - hence the timeout.
 */
function afterVideoFramePresented(callback) {
  let completed = false;
  let videoFrameCallback = null;
  const finish = () => {
    if (completed) return;
    completed = true;
    window.clearTimeout(fallbackTimer);
    callback();
  };
  const fallbackTimer = window.setTimeout(() => {
    if (videoFrameCallback !== null) {
      elements.video.cancelVideoFrameCallback(videoFrameCallback);
    }
    finish();
  }, FRAME_PRESENT_TIMEOUT_MS);
  if (typeof elements.video.requestVideoFrameCallback === "function") {
    videoFrameCallback = elements.video.requestVideoFrameCallback(finish);
  }
}
