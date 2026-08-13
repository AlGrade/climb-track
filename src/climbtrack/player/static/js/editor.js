/** Editing and persisting move boundaries. */

import { saveMoves } from "./api.js";
import { elements } from "./dom.js";
import { renderDraft, showValidationMessage } from "./editor-view.js";
import { requestRender } from "./render.js";
import { setSelectedMove } from "./selection.js";
import { state } from "./state.js";
import { setSaveState } from "./status.js";
import { hasFrame, lastFrameIdx, nearestFrame } from "./timeline.js";

/** Take a boundary from the current video position. */
export function setBoundaryFromVideo(name) {
  const frame = nearestFrame();
  if (!frame) return;
  state.selectionFollowsTimeline = false;
  state.draft[`${name}_frame`] = frame.frame_idx;
  renderDraft();
}

/** Take a boundary from the typed frame number. */
export function setBoundaryFromInput(name, input) {
  state.selectionFollowsTimeline = false;
  const raw = input.value.trim();
  if (raw === "") {
    state.draft[`${name}_frame`] = null;
    renderDraft();
    return;
  }
  const parsed = Number.parseInt(raw, 10);
  if (!hasFrame(parsed)) {
    state.draft[`${name}_frame`] = null;
    renderDraft();
    showValidationMessage(`Frame ${raw} is outside this video (0–${lastFrameIdx()}).`);
    return;
  }
  state.draft[`${name}_frame`] = parsed;
  renderDraft();
}

export function setDraftHand(hand) {
  state.selectionFollowsTimeline = false;
  state.draft.moving_hand = hand;
  renderDraft();
}

export async function saveDraft() {
  state.selectionFollowsTimeline = false;
  const edit = {
    start_frame: state.draft.start_frame,
    end_frame: state.draft.end_frame,
    moving_hand: state.draft.moving_hand,
    confidence: 1,
    source: state.selectedIndex >= 0 ? "corrected" : "manual",
    is_reviewed: true,
    outcome: state.draft.outcome,
  };
  const edits = state.session.moves.map(toEdit);
  if (state.selectedIndex >= 0) edits[state.selectedIndex] = edit;
  else edits.push(edit);
  await persistMoves(edits, edit);
}

export async function deleteSelected() {
  if (state.selectedIndex < 0) return;
  state.selectionFollowsTimeline = false;
  const edits = state.session.moves
    .filter((_, index) => index !== state.selectedIndex)
    .map(toEdit);
  await persistMoves(edits, null);
}

function toEdit(move) {
  return {
    move_id: move.move_id,
    start_frame: move.start_frame,
    end_frame: move.end_frame,
    moving_hand: move.moving_hand,
    confidence: move.confidence,
    source: move.source,
    is_reviewed: move.is_reviewed,
    outcome: move.outcome,
  };
}

/**
 * Send the whole move list and adopt the server's answer.
 *
 * Metrics are dropped on purpose: they describe the boundaries that just
 * changed, and the metric stage only runs when the player starts. Keeping the
 * old numbers would quietly attribute them to the new boundaries.
 */
async function persistMoves(edits, selectedEdit) {
  setSaveState("Saving…", "loading");
  elements.saveMove.disabled = true;
  try {
    const payload = await saveMoves(state.session.revision, edits);
    state.session = payload.session;
    state.metrics = [];
    state.speedTimeline = [];
    if (selectedEdit) {
      const savedIndex = state.session.moves.findIndex(
        (move) =>
          move.start_frame === selectedEdit.start_frame
          && move.end_frame === selectedEdit.end_frame
          && move.moving_hand === selectedEdit.moving_hand,
      );
      if (savedIndex >= 0) setSelectedMove(savedIndex);
      else state.selectedIndex = -1;
    } else {
      setSelectedMove(-1);
    }
    requestRender();
    setSaveState("Saved locally", "saved");
  } catch (error) {
    setSaveState("Save failed", "error");
    renderDraft();
    showValidationMessage(error.message);
  }
}
