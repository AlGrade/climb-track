/** Rendering of the boundary editor.
 *
 * Split from the editor's controls so that the selection module can refresh the
 * draft display without importing the editing logic that imports it back.
 */

import { elements, handButtons } from "./dom.js";
import { formatTime } from "./format.js";
import { state } from "./state.js";
import { lastFrameIdx, mediaTime } from "./timeline.js";

export function renderDraft() {
  const draft = state.draft;
  elements.editorTitle.textContent =
    state.selectedIndex >= 0 ? `Move ${state.selectedIndex + 1}` : "New move";
  elements.startValue.textContent = boundaryText(draft.start_frame);
  elements.endValue.textContent = boundaryText(draft.end_frame);
  syncBoundaryInput(elements.startFrameInput, draft.start_frame);
  syncBoundaryInput(elements.endFrameInput, draft.end_frame);
  handButtons.forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.hand === draft.moving_hand));
  });

  const complete =
    draft.start_frame !== null && draft.end_frame !== null && draft.moving_hand !== null;
  const forward = complete && draft.end_frame > draft.start_frame;
  elements.saveMove.disabled = !forward;
  elements.saveMove.textContent = "Save";
  elements.deleteMove.hidden = state.selectedIndex < 0;
  elements.validationMessage.textContent = draftBlockReason(draft, complete, forward);
}

function boundaryText(frameIdx) {
  if (frameIdx === null) return "Unset";
  return formatTime(mediaTime(frameIdx));
}

function syncBoundaryInput(input, frameIdx) {
  // Never overwrite the field the user is typing into.
  if (document.activeElement === input) return;
  input.value = frameIdx === null ? "" : String(frameIdx);
}

/** Name what is missing instead of leaving Save greyed out without explanation. */
function draftBlockReason(draft, complete, forward) {
  if (complete) return forward ? "" : "The end must be after the start.";
  const missing = [];
  if (draft.start_frame === null) missing.push("a start frame");
  if (draft.end_frame === null) missing.push("an end frame");
  if (draft.moving_hand === null) missing.push("the moving hand");
  const last = missing.pop();
  const listed = missing.length ? `${missing.join(", ")} and ${last}` : last;
  return `Saving needs ${listed}.`;
}

export function showValidationMessage(message) {
  elements.validationMessage.textContent = message;
}

/** Bound the frame inputs once the timeline is known. */
export function syncBoundaryLimits() {
  elements.startFrameInput.max = String(lastFrameIdx());
  elements.endFrameInput.max = String(lastFrameIdx());
}
