/** Which move is selected, and when the video position is allowed to change it. */

import { HAND_LABELS, OUTCOME_LABELS } from "./constants.js";
import { elements } from "./dom.js";
import { renderDraft } from "./editor-view.js";
import { renderMetrics } from "./metrics.js";
import { renderMoveList, updateNavigation } from "./move-list.js";
import { requestRender } from "./render.js";
import { emptyDraft, state } from "./state.js";

/** Select a move and load it into the editor draft. Pass -1 to select nothing. */
export function setSelectedMove(index) {
  state.selectedIndex = index;
  if (index < 0) {
    state.draft = emptyDraft();
    elements.activeMove.textContent = "Full video";
    return;
  }
  const move = state.session.moves[index];
  state.draft = {
    start_frame: move.start_frame,
    end_frame: move.end_frame,
    moving_hand: move.moving_hand,
    outcome: move.outcome,
  };
  elements.activeMove.textContent =
    `Move ${move.move_id} · ${HAND_LABELS[move.moving_hand]} · ${OUTCOME_LABELS[move.outcome]}`;
}

/** A deliberate pick by the user, which also stops the selection from following the video. */
export function selectMove(index) {
  if (!state.session.moves.length) return;
  state.selectionFollowsTimeline = false;
  setSelectedMove(Math.max(0, Math.min(index, state.session.moves.length - 1)));
  requestRender();
}

export function clearSelection() {
  state.selectionFollowsTimeline = false;
  setSelectedMove(-1);
  requestRender();
}

/**
 * Follow the video position, unless the user is editing.
 *
 * Searching for a new boundary necessarily leaves the current move, and letting
 * the timeline reset the selection there would discard the draft being corrected
 * - which used to make exactly those corrections impossible to save.
 */
export function syncMoveSelection(frameIdx) {
  if (!state.selectionFollowsTimeline) return;
  if (elements.editorCard.open) return;
  const matchingIndex = moveIndexForFrame(frameIdx);
  if (matchingIndex === state.selectedIndex) return;
  setSelectedMove(matchingIndex);
  renderMoveList();
  renderDraft();
  renderMetrics();
  updateNavigation();
}

/** Index of the last move containing this frame, or -1. */
function moveIndexForFrame(frameIdx) {
  for (let index = state.session.moves.length - 1; index >= 0; index -= 1) {
    const move = state.session.moves[index];
    if (move.start_frame <= frameIdx && frameIdx <= move.end_frame) return index;
  }
  return -1;
}
