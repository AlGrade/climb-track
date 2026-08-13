/** Values that are fixed for the lifetime of the page. */

export const HAND_LABELS = Object.freeze({
  left: "Left hand",
  right: "Right hand",
  both: "Both hands",
});

export const OUTCOME_LABELS = Object.freeze({
  completed: "Completed",
  fall: "Fall",
});

/** SVG user units of the speed chart; must match the viewBox in index.html. */
export const CHART_LAYOUT = Object.freeze({
  width: 760,
  left: 58,
  right: 16,
  top: 18,
  bottom: 224,
});

export const LAYOUT_STORAGE_KEY = "climbtrack-player-layout";

/** How long the fullscreen overlay stays visible after the last input. */
export const CHROME_IDLE_DELAY_MS = 2200;

/** Upper bound on frame steps buffered while a seek is still in flight. */
export const MAX_QUEUED_FRAME_STEPS = 12;

/** Give up waiting for a presented video frame after this long. */
export const FRAME_PRESENT_TIMEOUT_MS = 100;
