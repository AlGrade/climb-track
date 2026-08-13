/** The small save-state banner under the editor. */

import { elements } from "./dom.js";

/** @param {"loading"|"saved"|"error"} status */
export function setSaveState(message, status) {
  elements.saveState.textContent = message;
  elements.saveState.dataset.state = status;
}
