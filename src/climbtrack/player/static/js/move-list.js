/** The list of detected moves, and the enabled state of the controls around it. */

import { HAND_LABELS, OUTCOME_LABELS } from "./constants.js";
import { elements } from "./dom.js";
import { formatTime } from "./format.js";
import { state } from "./state.js";
import { mediaTime } from "./timeline.js";

/**
 * Rows carry their index as a data attribute instead of a closure, so main.js can
 * handle clicks with one delegated listener that survives every re-render.
 */
export function renderMoveList() {
  elements.moveList.replaceChildren();
  const moves = state.session.moves;
  elements.emptyMoves.hidden = moves.length > 0;
  moves.forEach((move, index) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "move-row";
    row.dataset.moveIndex = String(index);
    row.setAttribute("aria-current", String(index === state.selectedIndex));
    if (move.outcome === "fall") row.classList.add("move-row-fall");

    const number = document.createElement("span");
    number.className = "move-number";
    number.textContent = `#${move.move_id}`;

    const summary = document.createElement("span");
    summary.className = "move-summary";
    const title = document.createElement("strong");
    title.textContent = `${HAND_LABELS[move.moving_hand]} · ${OUTCOME_LABELS[move.outcome]}`;
    const times = document.createElement("span");
    times.textContent =
      `${formatTime(mediaTime(move.start_frame))} – ${formatTime(mediaTime(move.end_frame))}`;
    summary.append(title, times);

    const duration = document.createElement("span");
    duration.className = "move-duration";
    duration.textContent = `${(move.end_timestamp - move.start_timestamp).toFixed(2)} s`;

    row.append(number, summary, duration);
    elements.moveList.append(row);
  });
}

/** Controls stay disabled until the video can actually be driven. */
export function updateNavigation() {
  const videoReady = elements.video.readyState > 0;
  const hasMoves = Boolean(state.session) && state.session.moves.length > 0 && videoReady;
  elements.previousMove.disabled = !hasMoves;
  elements.replayMove.disabled = !hasMoves;
  elements.nextMove.disabled = !hasMoves;
  elements.playAll.disabled = !videoReady;
  elements.videoToggle.disabled = !videoReady;
  elements.videoScrubber.disabled = !videoReady;
  elements.videoMute.disabled = !videoReady;
  elements.videoPreviousFrame.disabled = !videoReady;
  elements.videoNextFrame.disabled = !videoReady;
}
