/** Desktop layout switch, fullscreen, and the auto-hiding overlay chrome. */

import { CHROME_IDLE_DELAY_MS, LAYOUT_STORAGE_KEY } from "./constants.js";
import { elements } from "./dom.js";
import { state } from "./state.js";

// ------------------------------------------------------------------ layout

export function initializeLayout() {
  let storedLayout = null;
  try {
    storedLayout = window.localStorage.getItem(LAYOUT_STORAGE_KEY);
  } catch (error) {
    // Private browsing modes can refuse storage; the default layout still works.
    console.warn("Could not read the saved player layout", error);
  }
  applyLayout(storedLayout === "portrait" ? "portrait" : "landscape", { persist: false });
}

export function toggleLayout() {
  applyLayout(state.layoutMode === "portrait" ? "landscape" : "portrait");
}

/**
 * Landscape stacks video and chart; portrait gives a 9:16 video a window without
 * wide black bars and moves the transport directly below it.
 */
function applyLayout(layout, options = {}) {
  state.layoutMode = layout;
  document.documentElement.dataset.layout = layout;
  const portrait = layout === "portrait";
  arrangeLayout(layout);
  elements.layoutToggle.setAttribute("aria-pressed", String(portrait));
  elements.layoutToggle.setAttribute(
    "aria-label",
    portrait ? "Switch to landscape layout" : "Switch to portrait layout",
  );
  elements.layoutToggleIcon.textContent = portrait ? "▯" : "▭";
  elements.layoutToggleLabel.textContent = portrait ? "Portrait layout" : "Landscape layout";
  if (options.persist !== false) {
    try {
      window.localStorage.setItem(LAYOUT_STORAGE_KEY, layout);
    } catch (error) {
      console.warn("Could not save the player layout", error);
    }
  }
  window.requestAnimationFrame(() => onLayoutSettled());
}

function arrangeLayout(layout) {
  if (layout === "portrait") {
    elements.videoShell.after(elements.transport);
    elements.editorCard.after(elements.metricsCard);
  } else {
    elements.videoShell.after(elements.metricsCard);
    elements.movesCard.after(elements.transport);
  }
}

// Set by main.js so the layout can refresh the readout without importing playback.
let onLayoutSettled = () => {};

export function setLayoutSettledHandler(callback) {
  onLayoutSettled = callback;
}

// -------------------------------------------------------------- fullscreen

/**
 * Only the video shell goes fullscreen, so the chart, move list, and editor
 * disappear on their own while the overlay controls stay clickable.
 */
function isFullscreen() {
  const active = document.fullscreenElement || document.webkitFullscreenElement || null;
  return active === elements.videoShell;
}

export function toggleFullscreen() {
  try {
    if (isFullscreen()) {
      const exit = document.exitFullscreen || document.webkitExitFullscreen;
      Promise.resolve(exit.call(document)).catch(reportFullscreenError);
      return;
    }
    const request =
      elements.videoShell.requestFullscreen || elements.videoShell.webkitRequestFullscreen;
    if (!request) throw new Error("this browser offers no fullscreen mode");
    Promise.resolve(request.call(elements.videoShell)).catch(reportFullscreenError);
  } catch (error) {
    reportFullscreenError(error);
  }
}

function reportFullscreenError(error) {
  elements.activeMove.textContent = `Fullscreen failed: ${error.message}`;
  console.error("Fullscreen request failed", error);
}

export function syncFullscreenControls() {
  const active = isFullscreen();
  // The stylesheet reads this attribute instead of `:fullscreen`, so the
  // -webkit- prefixed fallback above styles the shell just like the standard API.
  document.documentElement.dataset.fullscreen = String(active);
  elements.fullscreenToggle.setAttribute("aria-pressed", String(active));
  elements.fullscreenToggle.setAttribute("aria-label", active ? "Exit fullscreen" : "Enter fullscreen");
  elements.fullscreenToggleIcon.textContent = active ? "⤡" : "⛶";
  elements.fullscreenToggleLabel.textContent = active ? "Exit fullscreen" : "Fullscreen";
  revealChrome();
  window.requestAnimationFrame(() => onLayoutSettled());
}

// ------------------------------------------------------------------ chrome

/**
 * The overlay chrome only ever hides during fullscreen playback. A paused video
 * keeps it, because that is exactly when frames are stepped and read.
 */
export function revealChrome() {
  // Guarded because every pointermove calls this; a redundant attribute write
  // would invalidate styles while a 1080p video is decoding.
  if (document.documentElement.dataset.chrome !== "visible") {
    document.documentElement.dataset.chrome = "visible";
  }
  window.clearTimeout(state.chromeIdleTimer);
  state.chromeIdleTimer = null;
  if (!isFullscreen() || elements.video.paused) return;
  state.chromeIdleTimer = window.setTimeout(hideChromeIfIdle, CHROME_IDLE_DELAY_MS);
}

function hideChromeIfIdle() {
  state.chromeIdleTimer = null;
  if (!isFullscreen() || elements.video.paused) return;
  const busy =
    elements.videoControls.matches(":hover")
    || elements.videoActions.matches(":hover")
    || (document.activeElement !== elements.video
      && elements.videoShell.contains(document.activeElement));
  if (busy) {
    state.chromeIdleTimer = window.setTimeout(hideChromeIfIdle, CHROME_IDLE_DELAY_MS);
    return;
  }
  document.documentElement.dataset.chrome = "hidden";
}
