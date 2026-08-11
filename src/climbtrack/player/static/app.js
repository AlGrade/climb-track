"use strict";

const elements = Object.fromEntries(
  [
    "video", "videoToggle", "videoScrubber", "videoTime", "videoMute", "activeMove",
    "saveState", "currentTime", "currentFrame", "previousMove",
    "replayMove", "nextMove", "playAll", "previousFrame", "nextFrame", "editorTitle", "resetDraft", "setStart",
    "setEnd", "startValue", "endValue", "saveMove", "deleteMove", "validationMessage",
    "emptyMoves", "moveList", "metricsCard", "metricsGrid", "metricsNote",
    "handMaxSpeed", "handMaxSpeedPx", "bodyMaxSpeed", "bodyMaxSpeedPx",
    "handMeanSpeed", "handPath", "bodyMeanSpeed", "bodyPath", "speedChartWrap",
    "handSpeedPath", "bodySpeedPath", "chartCursor", "chartGrid", "chartHandValue",
    "chartBodyValue", "chartFrameLabel",
  ].map((id) => [id, document.getElementById(id)]),
);

const state = {
  session: null,
  timeline: [],
  settings: null,
  metrics: [],
  speedTimeline: [],
  chartSamples: [],
  timelineIndexByFrame: new Map(),
  selectedIndex: -1,
  draft: { start_frame: null, end_frame: null, moving_hand: null, outcome: "completed" },
  playbackEnd: null,
  playbackReady: true,
  watcherRunning: false,
  scrubbing: false,
  scrubTargetIndex: null,
  scrubAppliedIndex: null,
  scrubAnimationFrame: null,
  scrubSeekInFlight: false,
  scrubFinishing: false,
  frameStepInFlight: false,
  queuedFrameSteps: 0,
  heldFrameDirection: 0,
  frameHoldTimer: null,
};

const handLabels = { left: "Left hand", right: "Right hand", both: "Both hands" };
const outcomeLabels = { completed: "Completed", fall: "Fall" };
const chartLayout = Object.freeze({ width: 760, left: 58, right: 16, top: 18, bottom: 224 });

async function initialize() {
  try {
    const response = await fetch("/api/session");
    if (!response.ok) throw new Error(`Could not load player data (${response.status})`);
    const payload = await response.json();
    state.session = payload.session;
    state.timeline = payload.timeline;
    state.timelineIndexByFrame = new Map(
      state.timeline.map((frame, index) => [frame.frame_idx, index]),
    );
    state.settings = payload.settings;
    state.metrics = payload.metrics || [];
    state.speedTimeline = payload.speed_timeline || [];
    elements.video.src = payload.video.url;
    renderAll();
    setSaveState("Saved locally", "saved");
  } catch (error) {
    setSaveState(error.message, "error");
  }
}

function renderAll() {
  renderMoveList();
  renderDraft();
  renderMetrics();
  updateReadout();
  updateNavigation();
}

function renderMetrics() {
  if (state.selectedIndex < 0 || !state.session.moves.length) {
    state.chartSamples = [];
    elements.metricsCard.hidden = true;
    return;
  }
  const move = state.session.moves[state.selectedIndex];
  const metrics = state.metrics.find((row) => row.move_id === move.move_id);
  elements.metricsCard.hidden = false;
  if (!metrics) {
    state.chartSamples = [];
    elements.metricsGrid.hidden = true;
    elements.speedChartWrap.hidden = true;
    elements.metricsNote.textContent = "Restart the player to recalculate.";
    return;
  }
  elements.metricsGrid.hidden = false;
  elements.speedChartWrap.hidden = false;
  elements.handMaxSpeed.textContent = formatRelativeSpeed(metrics.hand_max_speed_body_lengths_s);
  elements.handMaxSpeedPx.textContent = `${metrics.hand_max_speed_px_s.toFixed(0)} px/s`;
  elements.bodyMaxSpeed.textContent = formatRelativeSpeed(metrics.body_max_speed_body_lengths_s);
  elements.bodyMaxSpeedPx.textContent = `${metrics.body_max_speed_px_s.toFixed(0)} px/s`;
  elements.handMeanSpeed.textContent = formatRelativeSpeed(metrics.hand_mean_speed_body_lengths_s);
  elements.handPath.textContent = `Path ${metrics.hand_path_length_px.toFixed(0)} px`;
  elements.bodyMeanSpeed.textContent = formatRelativeSpeed(metrics.body_mean_speed_body_lengths_s);
  elements.bodyPath.textContent = `Path ${metrics.body_path_length_px.toFixed(0)} px`;
  elements.metricsNote.textContent = "BL/s = body lengths per second";
  renderSpeedChart(move);
}

function renderSpeedChart(move) {
  const samples = state.speedTimeline.filter((sample) => sample.move_id === move.move_id);
  state.chartSamples = samples;
  if (samples.length < 2) {
    elements.speedChartWrap.hidden = true;
    return;
  }
  const duration = samples[samples.length - 1].offset_seconds || 1;
  const maximum = Math.max(
    ...samples.map((sample) => sample.hand_speed_body_lengths_s),
    ...samples.map((sample) => sample.body_speed_body_lengths_s),
    0.01,
  );
  const scale = chartScale(maximum);
  const plotWidth = chartLayout.width - chartLayout.left - chartLayout.right;
  const plotHeight = chartLayout.bottom - chartLayout.top;
  const pathFor = (key) => samples.map((sample, index) => {
    const x = chartLayout.left + (sample.offset_seconds / duration) * plotWidth;
    const y = chartLayout.bottom - (sample[key] / scale.maximum) * plotHeight;
    return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ");
  renderChartGrid(duration, scale);
  elements.handSpeedPath.setAttribute("d", pathFor("hand_speed_body_lengths_s"));
  elements.bodySpeedPath.setAttribute("d", pathFor("body_speed_body_lengths_s"));
  updateChartCursor();
}

function chartScale(maximum) {
  const roughStep = maximum / 4;
  const magnitude = 10 ** Math.floor(Math.log10(roughStep));
  const normalized = roughStep / magnitude;
  let normalizedStep = 10;
  if (normalized <= 1.5) normalizedStep = 1;
  else if (normalized <= 3) normalizedStep = 2;
  else if (normalized <= 7) normalizedStep = 5;
  const step = normalizedStep * magnitude;
  const tickCount = Math.max(1, Math.ceil(maximum / step));
  return { maximum: step * tickCount, step, tickCount };
}

function renderChartGrid(duration, scale) {
  elements.chartGrid.replaceChildren();
  for (let index = 0; index <= scale.tickCount; index += 1) {
    const value = index * scale.step;
    const y = chartLayout.bottom - (value / scale.maximum) * (chartLayout.bottom - chartLayout.top);
    appendSvgElement("line", {
      class: `chart-grid${index === 0 ? " chart-grid-axis" : ""}`,
      x1: chartLayout.left,
      y1: y,
      x2: chartLayout.width - chartLayout.right,
      y2: y,
    });
    appendSvgElement("text", {
      class: "chart-tick-label",
      x: chartLayout.left - 10,
      y: y + 4,
      "text-anchor": "end",
    }, formatAxisNumber(value));
  }

  const xTickCount = 4;
  for (let index = 0; index <= xTickCount; index += 1) {
    const fraction = index / xTickCount;
    const x = chartLayout.left + fraction * (chartLayout.width - chartLayout.left - chartLayout.right);
    appendSvgElement("line", {
      class: `chart-grid${index === 0 ? " chart-grid-axis" : ""}`,
      x1: x,
      y1: chartLayout.top,
      x2: x,
      y2: chartLayout.bottom,
    });
    appendSvgElement("text", {
      class: "chart-tick-label",
      x,
      y: chartLayout.bottom + 23,
      "text-anchor": "middle",
    }, `${(duration * fraction).toFixed(1)} s`);
  }

  appendSvgElement("text", {
    class: "chart-axis-title",
    x: -(chartLayout.top + chartLayout.bottom) / 2,
    y: 13,
    transform: "rotate(-90)",
    "text-anchor": "middle",
  }, "Speed (BL/s)");
}

function appendSvgElement(name, attributes, text = null) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
  if (text !== null) element.textContent = text;
  elements.chartGrid.append(element);
}

function formatAxisNumber(value) {
  return Number(value.toFixed(2)).toString();
}

function updateChartCursor(mediaTimeValue = elements.video.currentTime) {
  if (state.selectedIndex < 0 || state.chartSamples.length < 2) {
    elements.chartCursor.setAttribute("visibility", "hidden");
    elements.chartHandValue.textContent = "–";
    elements.chartBodyValue.textContent = "–";
    elements.chartFrameLabel.textContent = "Frame –";
    return;
  }
  const move = state.session.moves[state.selectedIndex];
  const offset = mediaTimeValue - mediaTime(move.start_frame);
  const duration = state.chartSamples[state.chartSamples.length - 1].offset_seconds;
  const clampedOffset = Math.max(0, Math.min(offset, duration));
  const x = chartLayout.left
    + (clampedOffset / duration) * (chartLayout.width - chartLayout.left - chartLayout.right);
  const sample = nearestChartSample(clampedOffset);
  elements.chartCursor.setAttribute("visibility", "visible");
  elements.chartCursor.setAttribute("x1", x.toFixed(2));
  elements.chartCursor.setAttribute("x2", x.toFixed(2));
  elements.chartHandValue.textContent = formatRelativeSpeed(sample.hand_speed_body_lengths_s);
  elements.chartBodyValue.textContent = formatRelativeSpeed(sample.body_speed_body_lengths_s);
  elements.chartFrameLabel.textContent = `Frame ${sample.frame_idx}`;
}

function nearestChartSample(offset) {
  let low = 0;
  let high = state.chartSamples.length - 1;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (state.chartSamples[middle].offset_seconds < offset) low = middle + 1;
    else high = middle;
  }
  if (low === 0) return state.chartSamples[0];
  const before = state.chartSamples[low - 1];
  const after = state.chartSamples[low];
  return Math.abs(before.offset_seconds - offset) <= Math.abs(after.offset_seconds - offset)
    ? before
    : after;
}

function formatRelativeSpeed(value) {
  return `${value.toFixed(2)} BL/s`;
}

function renderMoveList() {
  elements.moveList.replaceChildren();
  const moves = state.session.moves;
  elements.emptyMoves.hidden = moves.length > 0;
  moves.forEach((move, index) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "move-row";
    row.setAttribute("aria-current", String(index === state.selectedIndex));
    row.addEventListener("click", () => selectMove(index, { seek: true }));

    const number = document.createElement("span");
    number.className = "move-number";
    number.textContent = `#${move.move_id}`;

    const summary = document.createElement("span");
    summary.className = "move-summary";
    const title = document.createElement("strong");
    title.textContent = `${handLabels[move.moving_hand]} · ${outcomeLabels[move.outcome]}`;
    if (move.outcome === "fall") row.classList.add("move-row-fall");
    const times = document.createElement("span");
    times.textContent = `${formatTime(mediaTime(move.start_frame))} – ${formatTime(mediaTime(move.end_frame))}`;
    summary.append(title, times);

    const duration = document.createElement("span");
    duration.className = "move-duration";
    duration.textContent = `${(move.end_timestamp - move.start_timestamp).toFixed(2)} s`;
    row.append(number, summary, duration);
    elements.moveList.append(row);
  });
}

function renderDraft() {
  const draft = state.draft;
  elements.editorTitle.textContent = state.selectedIndex >= 0
    ? `Move ${state.selectedIndex + 1}`
    : "New move";
  elements.startValue.textContent = boundaryText(draft.start_frame);
  elements.endValue.textContent = boundaryText(draft.end_frame);
  document.querySelectorAll("[data-hand]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.hand === draft.moving_hand));
  });
  const complete = draft.start_frame !== null && draft.end_frame !== null && draft.moving_hand !== null;
  const forward = complete && draft.end_frame > draft.start_frame;
  elements.saveMove.disabled = !forward;
  elements.saveMove.textContent = "Save";
  elements.deleteMove.hidden = state.selectedIndex < 0;
  elements.validationMessage.textContent = complete && !forward
    ? "The end must be after the start."
    : "";
}

function updateNavigation() {
  const videoReady = elements.video.readyState > 0;
  const hasMoves = state.session && state.session.moves.length > 0 && videoReady;
  elements.previousMove.disabled = !hasMoves;
  elements.replayMove.disabled = !hasMoves;
  elements.nextMove.disabled = !hasMoves;
  elements.playAll.disabled = !videoReady;
  elements.videoToggle.disabled = !videoReady;
  elements.videoScrubber.disabled = !videoReady;
  elements.videoMute.disabled = !videoReady;
}

function selectMove(index, options = {}) {
  if (!state.session.moves.length) return;
  state.selectedIndex = Math.max(0, Math.min(index, state.session.moves.length - 1));
  const move = state.session.moves[state.selectedIndex];
  state.draft = {
    start_frame: move.start_frame,
    end_frame: move.end_frame,
    moving_hand: move.moving_hand,
    outcome: move.outcome,
  };
  elements.activeMove.textContent = `Move ${move.move_id} · ${handLabels[move.moving_hand]} · ${outcomeLabels[move.outcome]}`;
  if (options.seek) seekToFrame(move.start_frame);
  renderAll();
}

function playMove(index) {
  if (!state.session.moves.length) return;
  selectMove(index, { seek: false });
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

function startPlayback(start, end) {
  cancelFrameStepping();
  state.playbackReady = false;
  state.playbackEnd = end;

  // Keep play() in the original click task, then seek once Chrome has accepted
  // playback. Seeking first can make Chrome interrupt the play request.
  elements.video.play()
    .then(() => {
      const needsSeek = Math.abs(elements.video.currentTime - start) >= 0.01 || elements.video.seeking;
      if (needsSeek) {
        elements.video.addEventListener("seeked", () => {
          state.playbackReady = true;
          window.requestAnimationFrame(() => {
            elements.video.play().catch(reportPlaybackError);
          });
        }, { once: true });
        elements.video.currentTime = start;
      } else {
        state.playbackReady = true;
      }
    })
    .catch(reportPlaybackError);
}

function reportPlaybackError(error) {
  state.playbackEnd = null;
  state.playbackReady = true;
  elements.activeMove.textContent = `Playback failed: ${error.message}`;
  console.error("Video playback failed", error);
}

function watchPlayback() {
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
    if (!elements.video.paused) {
      window.requestAnimationFrame(tick);
    } else {
      state.watcherRunning = false;
    }
  };
  window.requestAnimationFrame(tick);
}

function nearestTimelineIndex(time) {
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

function nearestFrame(time = elements.video.currentTime) {
  return state.timeline[nearestTimelineIndex(time)];
}

function mediaTime(frameIdx) {
  const timelineIndex = state.timelineIndexByFrame.get(frameIdx);
  const frame = timelineIndex === undefined ? null : state.timeline[timelineIndex];
  return frame ? frame.media_time : 0;
}

function seekToFrame(frameIdx) {
  const timelineIndex = state.timelineIndexByFrame.get(frameIdx);
  if (timelineIndex === undefined) return;
  seekToTimelineIndex(timelineIndex);
}

function seekToTimelineIndex(timelineIndex, options = {}) {
  if (!options.frameStep) cancelFrameStepping();
  elements.video.pause();
  state.playbackEnd = null;
  state.playbackReady = true;
  elements.video.currentTime = state.timeline[timelineIndex].media_time;
  updateReadoutForTimelineIndex(timelineIndex);
}

function stepFrame(offset) {
  if (!state.timeline.length) return;
  if (state.frameStepInFlight) {
    state.queuedFrameSteps = Math.max(-12, Math.min(12, state.queuedFrameSteps + offset));
    return;
  }
  const current = nearestTimelineIndex(elements.video.currentTime);
  const target = Math.max(0, Math.min(current + offset, state.timeline.length - 1));
  if (target === current) return;
  state.frameStepInFlight = true;
  seekToTimelineIndex(target, { frameStep: true });
}

function finishFrameStep() {
  if (!state.frameStepInFlight) return;
  state.frameStepInFlight = false;
  let nextDirection = state.heldFrameDirection;
  if (nextDirection === 0 && state.queuedFrameSteps !== 0) {
    nextDirection = Math.sign(state.queuedFrameSteps);
    state.queuedFrameSteps -= nextDirection;
  }
  if (nextDirection !== 0) {
    state.frameHoldTimer = window.setTimeout(() => {
      state.frameHoldTimer = null;
      stepFrame(nextDirection);
    }, 0);
  }
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

function setBoundary(name) {
  const frame = nearestFrame();
  if (!frame) return;
  state.draft[`${name}_frame`] = frame.frame_idx;
  renderDraft();
}

function boundaryText(frameIdx) {
  if (frameIdx === null) return "Unset";
  return `${formatTime(mediaTime(frameIdx))} · Frame ${frameIdx}`;
}

function resetDraft() {
  state.selectedIndex = -1;
  state.draft = { start_frame: null, end_frame: null, moving_hand: null, outcome: "completed" };
  elements.activeMove.textContent = "Full video";
  renderAll();
}

async function saveDraft() {
  const edit = {
    start_frame: state.draft.start_frame,
    end_frame: state.draft.end_frame,
    moving_hand: state.draft.moving_hand,
    confidence: 1,
    source: state.selectedIndex >= 0 ? "corrected" : "manual",
    is_reviewed: true,
    outcome: state.draft.outcome,
  };
  const edits = state.session.moves.map((move) => ({
    move_id: move.move_id,
    start_frame: move.start_frame,
    end_frame: move.end_frame,
    moving_hand: move.moving_hand,
    confidence: move.confidence,
    source: move.source,
    is_reviewed: move.is_reviewed,
    outcome: move.outcome,
  }));
  if (state.selectedIndex >= 0) edits[state.selectedIndex] = edit;
  else edits.push(edit);
  await persistMoves(edits, edit);
}

async function deleteSelected() {
  if (state.selectedIndex < 0) return;
  const edits = state.session.moves
    .filter((_, index) => index !== state.selectedIndex)
    .map((move) => ({
      move_id: move.move_id,
      start_frame: move.start_frame,
      end_frame: move.end_frame,
      moving_hand: move.moving_hand,
      confidence: move.confidence,
      source: move.source,
      is_reviewed: move.is_reviewed,
      outcome: move.outcome,
    }));
  await persistMoves(edits, null);
}

async function persistMoves(edits, selectedEdit) {
  setSaveState("Saving…", "loading");
  elements.saveMove.disabled = true;
  try {
    const response = await fetch("/api/moves", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ expected_revision: state.session.revision, moves: edits }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `Save failed (${response.status})`);
    state.session = payload.session;
    state.metrics = [];
    state.speedTimeline = [];
    if (selectedEdit) {
      state.selectedIndex = state.session.moves.findIndex((move) =>
        move.start_frame === selectedEdit.start_frame
        && move.end_frame === selectedEdit.end_frame
        && move.moving_hand === selectedEdit.moving_hand,
      );
      if (state.selectedIndex >= 0) {
        const move = state.session.moves[state.selectedIndex];
        state.draft = {
          start_frame: move.start_frame,
          end_frame: move.end_frame,
          moving_hand: move.moving_hand,
          outcome: move.outcome,
        };
        elements.activeMove.textContent = `Move ${move.move_id} · ${handLabels[move.moving_hand]} · ${outcomeLabels[move.outcome]}`;
      }
    } else {
      resetDraft();
    }
    renderAll();
    setSaveState("Saved locally", "saved");
  } catch (error) {
    setSaveState("Save failed", "error");
    renderDraft();
    elements.validationMessage.textContent = error.message;
  }
}

function updateReadout() {
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
  const progress = state.timeline.length <= 1
    ? 0
    : (timelineIndex / (state.timeline.length - 1)) * 100;
  elements.videoScrubber.style.setProperty("--scrub-progress", `${progress.toFixed(3)}%`);
  const duration = Number.isFinite(elements.video.duration)
    ? elements.video.duration
    : state.timeline[state.timeline.length - 1].media_time;
  elements.videoTime.textContent = `${formatControlTime(frame.media_time)} / ${formatControlTime(duration)}`;
  updateChartCursor(frame.media_time);
}

function configureVideoControls() {
  elements.videoScrubber.max = String(Math.max(0, state.timeline.length - 1));
  updateNavigation();
  updatePlaybackControls();
  updateReadout();
}

function updatePlaybackControls() {
  const playing = !elements.video.paused && !elements.video.ended;
  elements.videoToggle.textContent = playing ? "❚❚" : "▶";
  elements.videoToggle.setAttribute("aria-label", playing ? "Pause video" : "Play video");
  elements.videoMute.textContent = elements.video.muted ? "🔇" : "🔊";
  elements.videoMute.setAttribute("aria-label", elements.video.muted ? "Unmute video" : "Mute video");
}

function toggleVideoPlayback() {
  cancelFrameStepping();
  state.playbackEnd = null;
  if (elements.video.paused) {
    if (elements.video.ended) startPlayback(0, null);
    else elements.video.play().catch(reportPlaybackError);
  } else {
    elements.video.pause();
  }
}

function beginScrub() {
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

function previewScrub() {
  if (!state.scrubbing) beginScrub();
  state.scrubTargetIndex = Number(elements.videoScrubber.value);
  updateReadoutForTimelineIndex(state.scrubTargetIndex);
  scheduleScrubSeek();
}

function scheduleScrubSeek() {
  if (
    state.scrubAnimationFrame !== null
    || state.scrubSeekInFlight
    || state.scrubTargetIndex === null
  ) return;
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

function finishScrub() {
  if (!state.scrubbing) return;
  state.scrubTargetIndex ??= Number(elements.videoScrubber.value);
  state.scrubFinishing = true;
  scheduleScrubSeek();
  completeScrubIfReady();
}

function completeScrubIfReady() {
  if (
    !state.scrubFinishing
    || state.scrubSeekInFlight
    || state.scrubAnimationFrame !== null
    || state.scrubTargetIndex !== state.scrubAppliedIndex
  ) return;
  const target = state.scrubTargetIndex;
  state.scrubbing = false;
  state.scrubTargetIndex = null;
  state.scrubAppliedIndex = null;
  state.scrubFinishing = false;
  updateReadoutForTimelineIndex(target);
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds)) return "00:00.000";
  const minutes = Math.floor(seconds / 60);
  const remaining = seconds - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remaining.toFixed(3).padStart(6, "0")}`;
}

function formatControlTime(seconds) {
  if (!Number.isFinite(seconds)) return "00:00";
  const minutes = Math.floor(seconds / 60);
  const remaining = Math.floor(seconds - minutes * 60);
  return `${String(minutes).padStart(2, "0")}:${String(remaining).padStart(2, "0")}`;
}

function setSaveState(message, status) {
  elements.saveState.textContent = message;
  elements.saveState.dataset.state = status;
}

function handleVideoSeeked() {
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
  }, 100);
  if (typeof elements.video.requestVideoFrameCallback === "function") {
    videoFrameCallback = elements.video.requestVideoFrameCallback(finish);
  }
}

elements.video.addEventListener("play", () => {
  updatePlaybackControls();
  watchPlayback();
});
elements.video.addEventListener("pause", updatePlaybackControls);
elements.video.addEventListener("ended", updatePlaybackControls);
elements.video.addEventListener("volumechange", updatePlaybackControls);
elements.video.addEventListener("timeupdate", updateReadout);
elements.video.addEventListener("seeking", updateReadout);
elements.video.addEventListener("seeked", handleVideoSeeked);
elements.video.addEventListener("loadedmetadata", configureVideoControls);
elements.video.addEventListener("click", toggleVideoPlayback);
elements.videoToggle.addEventListener("click", toggleVideoPlayback);
elements.videoMute.addEventListener("click", () => {
  elements.video.muted = !elements.video.muted;
});
elements.videoScrubber.addEventListener("pointerdown", beginScrub);
elements.videoScrubber.addEventListener("input", previewScrub);
elements.videoScrubber.addEventListener("pointerup", finishScrub);
elements.videoScrubber.addEventListener("change", finishScrub);
elements.videoScrubber.addEventListener("pointercancel", finishScrub);
elements.previousMove.addEventListener("click", () => playMove(state.selectedIndex <= 0 ? 0 : state.selectedIndex - 1));
elements.replayMove.addEventListener("click", () => playMove(state.selectedIndex < 0 ? 0 : state.selectedIndex));
elements.nextMove.addEventListener("click", () => playMove(state.selectedIndex < 0 ? 0 : Math.min(state.selectedIndex + 1, state.session.moves.length - 1)));
elements.playAll.addEventListener("click", () => {
  state.selectedIndex = -1;
  elements.activeMove.textContent = "Full video";
  renderAll();
  startPlayback(0, null);
});
elements.previousFrame.addEventListener("click", () => stepFrame(-1));
elements.nextFrame.addEventListener("click", () => stepFrame(1));
elements.setStart.addEventListener("click", () => setBoundary("start"));
elements.setEnd.addEventListener("click", () => setBoundary("end"));
elements.resetDraft.addEventListener("click", resetDraft);
elements.saveMove.addEventListener("click", saveDraft);
elements.deleteMove.addEventListener("click", deleteSelected);
document.querySelectorAll("[data-hand]").forEach((button) => {
  button.addEventListener("click", () => {
    state.draft.moving_hand = button.dataset.hand;
    renderDraft();
  });
});

document.addEventListener("keydown", (event) => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) return;
  if (event.code === "Space") {
    event.preventDefault();
    if (!event.repeat) toggleVideoPlayback();
  } else if (event.key === "ArrowLeft") {
    event.preventDefault();
    if (!event.repeat) {
      state.heldFrameDirection = -1;
      stepFrame(-1);
    }
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    if (!event.repeat) {
      state.heldFrameDirection = 1;
      stepFrame(1);
    }
  }
});

document.addEventListener("keyup", (event) => {
  const direction = event.key === "ArrowLeft" ? -1 : event.key === "ArrowRight" ? 1 : 0;
  if (direction === 0 || state.heldFrameDirection !== direction) return;
  state.heldFrameDirection = 0;
  if (state.frameHoldTimer !== null) {
    window.clearTimeout(state.frameHoldTimer);
    state.frameHoldTimer = null;
  }
});

initialize();
