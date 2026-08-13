/** One lookup of every element the player touches, grouped by the area it belongs to.
 *
 * Resolved once at load time so the rest of the code never queries the document
 * again, and a missing id fails immediately instead of surfacing as a confusing
 * "cannot read properties of null" during playback.
 */

const IDS = [
  // Video shell and its overlay controls
  "videoShell", "video", "videoActions", "videoControls", "activeMove",
  "videoToggle", "videoScrubber", "videoTime", "videoMute",
  "videoPreviousFrame", "videoNextFrame",
  "layoutToggle", "layoutToggleIcon", "layoutToggleLabel",
  "fullscreenToggle", "fullscreenToggleIcon", "fullscreenToggleLabel",

  // Move list and transport
  "movesCard", "moveList", "emptyMoves", "transport",
  "previousMove", "replayMove", "nextMove", "playAll",

  // Boundary editor
  "editorCard", "editorTitle", "resetDraft", "currentTime", "currentFrame",
  "previousFrame", "nextFrame", "setStart", "setEnd",
  "startValue", "endValue", "startFrameInput", "endFrameInput",
  "saveMove", "deleteMove", "validationMessage", "saveState",

  // Metric tiles
  "metricsCard", "metricsGrid", "metricsNote",
  "handMaxSpeed", "handMaxSpeedPx", "bodyMaxSpeed", "bodyMaxSpeedPx",
  "handMeanSpeed", "handPath", "bodyMeanSpeed", "bodyPath",
  "hipRise", "hipRiseNote", "hipBelowHand", "hipBelowHandNote",
  "torsoLead", "torsoLeadNote", "handSettle", "handSettleNote",

  // Speed chart
  "speedChartWrap", "handSpeedPath", "bodySpeedPath", "chartCursor", "chartGrid",
  "chartHandValue", "chartBodyValue", "chartFrameLabel",
];

function lookup(id) {
  const element = document.getElementById(id);
  if (element === null) throw new Error(`The player markup is missing #${id}`);
  return element;
}

export const elements = Object.freeze(
  Object.fromEntries(IDS.map((id) => [id, lookup(id)])),
);

/** The moving-hand buttons, which are selected by attribute rather than by id. */
export const handButtons = Array.from(document.querySelectorAll("[data-hand]"));
