"use strict";

const token = new URLSearchParams(window.location.search).get("token") || "";
const elements = Object.fromEntries(
  [
    "video", "videoName", "activeMove", "saveState", "currentTime", "currentFrame",
    "moveCount", "previousMove", "replayMove", "nextMove", "playAll", "previousFrame",
    "nextFrame", "togglePlayback", "playbackRate", "editorTitle", "resetDraft", "setStart",
    "setEnd", "startValue", "endValue", "saveMove", "deleteMove", "validationMessage",
    "emptyMoves", "moveList",
  ].map((id) => [id, document.getElementById(id)]),
);

const state = {
  session: null,
  timeline: [],
  settings: null,
  selectedIndex: -1,
  draft: { start_frame: null, end_frame: null, moving_hand: null, outcome: "completed" },
  playbackEnd: null,
  watcherRunning: false,
};

const handLabels = { left: "Linke Hand", right: "Rechte Hand", both: "Beide Hände" };
const outcomeLabels = { completed: "Geschafft", fall: "Sturz" };

async function initialize() {
  if (!token) {
    setSaveState("Ungültiger Player-Link", "error");
    return;
  }
  try {
    const response = await fetch(`/api/session?token=${encodeURIComponent(token)}`);
    if (!response.ok) throw new Error(`Player-Daten konnten nicht geladen werden (${response.status})`);
    const payload = await response.json();
    state.session = payload.session;
    state.timeline = payload.timeline;
    state.settings = payload.settings;
    elements.videoName.textContent = payload.video.name;
    elements.video.src = payload.video.url;
    elements.video.playbackRate = Number(elements.playbackRate.value);
    renderAll();
    setSaveState("Lokal gespeichert", "saved");
  } catch (error) {
    setSaveState(error.message, "error");
  }
}

function renderAll() {
  renderMoveList();
  renderDraft();
  updateReadout();
  updateNavigation();
}

function renderMoveList() {
  elements.moveList.replaceChildren();
  const moves = state.session.moves;
  elements.moveCount.textContent = String(moves.length);
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
    ? `Zug ${state.selectedIndex + 1} bearbeiten`
    : "Neuen Zug markieren";
  elements.startValue.textContent = boundaryText(draft.start_frame);
  elements.endValue.textContent = boundaryText(draft.end_frame);
  document.querySelectorAll("[data-hand]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.hand === draft.moving_hand));
  });
  const complete = draft.start_frame !== null && draft.end_frame !== null && draft.moving_hand !== null;
  const forward = complete && draft.end_frame > draft.start_frame;
  elements.saveMove.disabled = !forward;
  elements.saveMove.textContent = state.selectedIndex >= 0 ? "Änderungen speichern" : "Zug speichern";
  elements.deleteMove.hidden = state.selectedIndex < 0;
  elements.validationMessage.textContent = complete && !forward
    ? "Das Ende muss nach dem Start liegen."
    : "";
}

function updateNavigation() {
  const hasMoves = state.session && state.session.moves.length > 0;
  elements.previousMove.disabled = !hasMoves;
  elements.replayMove.disabled = !hasMoves;
  elements.nextMove.disabled = !hasMoves;
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
  elements.activeMove.textContent = `Zug ${move.move_id} · ${handLabels[move.moving_hand]} · ${outcomeLabels[move.outcome]}`;
  if (options.seek) seekToFrame(move.start_frame);
  renderAll();
}

async function playMove(index) {
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
  elements.video.currentTime = start;
  state.playbackEnd = end;
  await elements.video.play();
}

function watchPlayback() {
  if (state.watcherRunning) return;
  state.watcherRunning = true;
  const tick = () => {
    updateReadout();
    if (state.playbackEnd !== null && elements.video.currentTime >= state.playbackEnd) {
      elements.video.pause();
      elements.video.currentTime = state.playbackEnd;
      state.playbackEnd = null;
      setSaveState("Zugende erreicht", "saved");
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

function nearestFrame() {
  return state.timeline[nearestTimelineIndex(elements.video.currentTime)];
}

function mediaTime(frameIdx) {
  const frame = state.timeline[frameIdx];
  return frame ? frame.media_time : 0;
}

function seekToFrame(frameIdx) {
  elements.video.pause();
  state.playbackEnd = null;
  elements.video.currentTime = mediaTime(frameIdx);
  updateReadout();
}

function stepFrame(offset) {
  elements.video.pause();
  state.playbackEnd = null;
  const target = Math.max(0, Math.min(nearestTimelineIndex(elements.video.currentTime) + offset, state.timeline.length - 1));
  seekToFrame(state.timeline[target].frame_idx);
}

function setBoundary(name) {
  const frame = nearestFrame();
  if (!frame) return;
  state.draft[`${name}_frame`] = frame.frame_idx;
  renderDraft();
}

function boundaryText(frameIdx) {
  if (frameIdx === null) return "Noch offen";
  return `${formatTime(mediaTime(frameIdx))} · Frame ${frameIdx}`;
}

function resetDraft() {
  state.selectedIndex = -1;
  state.draft = { start_frame: null, end_frame: null, moving_hand: null, outcome: "completed" };
  elements.activeMove.textContent = "Gesamtvideo";
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
  setSaveState("Wird gespeichert …", "loading");
  elements.saveMove.disabled = true;
  try {
    const response = await fetch(`/api/moves?token=${encodeURIComponent(token)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json", "X-ClimbTrack-Token": token },
      body: JSON.stringify({ expected_revision: state.session.revision, moves: edits }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `Speichern fehlgeschlagen (${response.status})`);
    state.session = payload.session;
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
        elements.activeMove.textContent = `Zug ${move.move_id} · ${handLabels[move.moving_hand]} · ${outcomeLabels[move.outcome]}`;
      }
    } else {
      resetDraft();
    }
    renderAll();
    setSaveState("Lokal gespeichert", "saved");
  } catch (error) {
    setSaveState("Speichern fehlgeschlagen", "error");
    renderDraft();
    elements.validationMessage.textContent = error.message;
  }
}

function updateReadout() {
  if (!state.timeline.length) return;
  const frame = nearestFrame();
  elements.currentTime.textContent = formatTime(elements.video.currentTime);
  elements.currentFrame.textContent = frame ? String(frame.frame_idx) : "–";
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds)) return "00:00.000";
  const minutes = Math.floor(seconds / 60);
  const remaining = seconds - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remaining.toFixed(3).padStart(6, "0")}`;
}

function setSaveState(message, status) {
  elements.saveState.textContent = message;
  elements.saveState.dataset.state = status;
}

elements.video.addEventListener("play", watchPlayback);
elements.video.addEventListener("timeupdate", updateReadout);
elements.previousMove.addEventListener("click", () => playMove(state.selectedIndex <= 0 ? 0 : state.selectedIndex - 1));
elements.replayMove.addEventListener("click", () => playMove(state.selectedIndex < 0 ? 0 : state.selectedIndex));
elements.nextMove.addEventListener("click", () => playMove(state.selectedIndex < 0 ? 0 : Math.min(state.selectedIndex + 1, state.session.moves.length - 1)));
elements.playAll.addEventListener("click", async () => {
  state.playbackEnd = null;
  state.selectedIndex = -1;
  elements.activeMove.textContent = "Gesamtvideo";
  elements.video.currentTime = 0;
  renderMoveList();
  await elements.video.play();
});
elements.previousFrame.addEventListener("click", () => stepFrame(-1));
elements.nextFrame.addEventListener("click", () => stepFrame(1));
elements.togglePlayback.addEventListener("click", () => {
  state.playbackEnd = null;
  if (elements.video.paused) elements.video.play();
  else elements.video.pause();
});
elements.playbackRate.addEventListener("change", () => {
  elements.video.playbackRate = Number(elements.playbackRate.value);
});
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
    elements.togglePlayback.click();
  } else if (event.key === "ArrowLeft") {
    event.preventDefault();
    stepFrame(-1);
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    stepFrame(1);
  }
});

initialize();
