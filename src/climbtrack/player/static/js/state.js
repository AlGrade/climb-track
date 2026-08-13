/** The single mutable store for the page.
 *
 * Controllers mutate it and then ask for a render; render functions only read it
 * and write to the DOM. Keeping that direction one-way is what stops the modules
 * from having to call each other in circles.
 */

export function emptyDraft() {
  return { start_frame: null, end_frame: null, moving_hand: null, outcome: "completed" };
}

export const state = {
  // Data from the server
  session: null,
  timeline: [],
  timelineIndexByFrame: new Map(),
  settings: null,
  metrics: [],
  speedTimeline: [],

  // What the user is looking at
  selectedIndex: -1,
  chartSamples: [],
  layoutMode: "landscape",

  // Whether the video position drives the selection. Turned on by full-video
  // playback and off as soon as the user picks or edits a move by hand.
  selectionFollowsTimeline: false,

  // The boundary editor's working copy of the selected move
  draft: emptyDraft(),

  // Playback bookkeeping
  playbackEnd: null,
  playbackReady: true,
  watcherRunning: false,

  // Scrubbing: the target index is what the user asked for, the applied index is
  // what has actually been handed to the video element.
  scrubbing: false,
  scrubTargetIndex: null,
  scrubAppliedIndex: null,
  scrubAnimationFrame: null,
  scrubSeekInFlight: false,
  scrubFinishing: false,

  // Single-frame stepping, which has to wait for each frame to be presented
  frameStepInFlight: false,
  queuedFrameSteps: 0,
  heldFrameDirection: 0,
  frameHoldTimer: null,

  // Fullscreen overlay auto-hide
  chromeIdleTimer: null,
};
