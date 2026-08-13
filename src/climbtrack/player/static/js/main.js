/** Composition root: wires the modules together and starts the player. */

import { fetchSession } from "./api.js";
import { elements, handButtons } from "./dom.js";
import { renderDraft, syncBoundaryLimits } from "./editor-view.js";
import {
  deleteSelected,
  saveDraft,
  setBoundaryFromInput,
  setBoundaryFromVideo,
  setDraftHand,
} from "./editor.js";
import {
  initializeLayout,
  revealChrome,
  setLayoutSettledHandler,
  syncFullscreenControls,
  toggleFullscreen,
  toggleLayout,
} from "./layout.js";
import { renderMetrics } from "./metrics.js";
import { renderMoveList, updateNavigation } from "./move-list.js";
import {
  beginScrub,
  configureVideoControls,
  finishScrub,
  handleVideoSeeked,
  holdFrameDirection,
  playMove,
  previewScrub,
  releaseFrameDirection,
  seekToFrame,
  seekToSelectedMoveStart,
  startPlayback,
  stepFrame,
  toggleVideoPlayback,
  updatePlaybackControls,
  updateReadout,
  watchPlayback,
} from "./playback.js";
import { setRenderer } from "./render.js";
import { clearSelection, selectMove, setSelectedMove } from "./selection.js";
import { hasFrame, loadTimeline } from "./timeline.js";
import { state } from "./state.js";
import { setSaveState } from "./status.js";

/** One full pass over every view. Controllers reach it through requestRender(). */
function renderAll() {
  renderMoveList();
  renderDraft();
  renderMetrics();
  updateReadout();
  updateNavigation();
}

function wireVideoElement() {
  elements.video.addEventListener("play", () => {
    updatePlaybackControls();
    watchPlayback();
    revealChrome();
  });
  elements.video.addEventListener("pause", () => {
    updatePlaybackControls();
    revealChrome();
  });
  elements.video.addEventListener("ended", () => {
    updatePlaybackControls();
    revealChrome();
  });
  elements.video.addEventListener("volumechange", updatePlaybackControls);
  elements.video.addEventListener("timeupdate", updateReadout);
  elements.video.addEventListener("seeking", updateReadout);
  elements.video.addEventListener("seeked", handleVideoSeeked);
  elements.video.addEventListener("loadedmetadata", () => {
    configureVideoControls();
    syncBoundaryLimits();
  });
  elements.video.addEventListener("click", toggleVideoPlayback);
}

function wireVideoControls() {
  elements.videoToggle.addEventListener("click", toggleVideoPlayback);
  elements.videoMute.addEventListener("click", () => {
    elements.video.muted = !elements.video.muted;
  });
  elements.videoScrubber.addEventListener("pointerdown", beginScrub);
  elements.videoScrubber.addEventListener("input", previewScrub);
  elements.videoScrubber.addEventListener("pointerup", finishScrub);
  elements.videoScrubber.addEventListener("change", finishScrub);
  elements.videoScrubber.addEventListener("pointercancel", finishScrub);
  elements.videoPreviousFrame.addEventListener("click", () => stepFrame(-1));
  elements.videoNextFrame.addEventListener("click", () => stepFrame(1));
}

function wireLayoutControls() {
  elements.layoutToggle.addEventListener("click", toggleLayout);
  elements.fullscreenToggle.addEventListener("click", toggleFullscreen);
  elements.videoShell.addEventListener("pointermove", revealChrome);
  elements.videoShell.addEventListener("pointerdown", revealChrome);
  document.addEventListener("fullscreenchange", syncFullscreenControls);
  document.addEventListener("webkitfullscreenchange", syncFullscreenControls);
}

function wireMoveNavigation() {
  // One delegated listener, so re-rendering the list cannot leave handlers behind.
  elements.moveList.addEventListener("click", (event) => {
    const row = event.target.closest("[data-move-index]");
    if (!row) return;
    selectMove(Number(row.dataset.moveIndex));
    seekToSelectedMoveStart();
  });
  elements.previousMove.addEventListener("click", () => {
    playMove(state.selectedIndex <= 0 ? 0 : state.selectedIndex - 1);
  });
  elements.replayMove.addEventListener("click", () => {
    playMove(state.selectedIndex < 0 ? 0 : state.selectedIndex);
  });
  elements.nextMove.addEventListener("click", () => {
    const last = state.session.moves.length - 1;
    playMove(state.selectedIndex < 0 ? 0 : Math.min(state.selectedIndex + 1, last));
  });
  elements.playAll.addEventListener("click", () => {
    state.selectionFollowsTimeline = true;
    setSelectedMove(-1);
    renderAll();
    startPlayback(0, null);
  });
}

function wireEditor() {
  elements.previousFrame.addEventListener("click", () => stepFrame(-1));
  elements.nextFrame.addEventListener("click", () => stepFrame(1));
  elements.setStart.addEventListener("click", () => setBoundaryFromVideo("start"));
  elements.setEnd.addEventListener("click", () => setBoundaryFromVideo("end"));
  [
    ["start", elements.startFrameInput],
    ["end", elements.endFrameInput],
  ].forEach(([name, input]) => {
    input.addEventListener("input", () => setBoundaryFromInput(name, input));
    input.addEventListener("keydown", (event) => {
      if (event.key !== "Enter") return;
      // Seek to the typed frame so it can be checked before saving.
      event.preventDefault();
      const frameIdx = Number.parseInt(input.value.trim(), 10);
      if (hasFrame(frameIdx)) seekToFrame(frameIdx);
    });
  });
  handButtons.forEach((button) => {
    button.addEventListener("click", () => setDraftHand(button.dataset.hand));
  });
  elements.resetDraft.addEventListener("click", clearSelection);
  elements.saveMove.addEventListener("click", saveDraft);
  elements.deleteMove.addEventListener("click", deleteSelected);
}

function wireKeyboard() {
  document.addEventListener("keydown", (event) => {
    // Never hijack keys while a value is being typed into a field.
    if (["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) return;
    revealChrome();
    if (event.code === "Space") {
      event.preventDefault();
      if (!event.repeat) toggleVideoPlayback();
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      if (!event.repeat) {
        holdFrameDirection(-1);
        stepFrame(-1);
      }
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      if (!event.repeat) {
        holdFrameDirection(1);
        stepFrame(1);
      }
    } else if (event.key === "f" || event.key === "F") {
      event.preventDefault();
      if (!event.repeat) toggleFullscreen();
    }
  });

  document.addEventListener("keyup", (event) => {
    if (event.key === "ArrowLeft") releaseFrameDirection(-1);
    else if (event.key === "ArrowRight") releaseFrameDirection(1);
  });
}

async function loadSession() {
  try {
    const payload = await fetchSession();
    state.session = payload.session;
    loadTimeline(payload.timeline);
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

function start() {
  setRenderer(renderAll);
  setLayoutSettledHandler(updateReadout);
  wireVideoElement();
  wireVideoControls();
  wireLayoutControls();
  wireMoveNavigation();
  wireEditor();
  wireKeyboard();
  initializeLayout();
  syncFullscreenControls();
  loadSession();
}

start();
