/** The per-frame speed chart and its cursor. */

import { CHART_LAYOUT } from "./constants.js";
import { elements } from "./dom.js";
import { formatAxisNumber, formatRelativeSpeed } from "./format.js";
import { state } from "./state.js";
import { mediaTime } from "./timeline.js";

const SVG_NAMESPACE = "http://www.w3.org/2000/svg";
const X_TICK_COUNT = 4;

/** Draw both speed curves for one move; hide the chart if it has too few samples. */
export function renderSpeedChart(move) {
  const samples = state.speedTimeline.filter((sample) => sample.move_id === move.move_id);
  state.chartSamples = samples;
  if (samples.length < 2) {
    elements.speedChartWrap.hidden = true;
    return;
  }
  const duration = samples[samples.length - 1].offset_seconds || 1;
  const maximum = Math.max(
    ...samples.map((sample) => sample.hand_speed_body_lengths_s),
    ...samples.map((sample) => sample.body_speed_body_lengths_s),
    0.01,
  );
  const scale = chartScale(maximum);
  renderChartGrid(duration, scale);
  elements.handSpeedPath.setAttribute("d", speedPath(samples, "hand_speed_body_lengths_s", duration, scale));
  elements.bodySpeedPath.setAttribute("d", speedPath(samples, "body_speed_body_lengths_s", duration, scale));
  updateChartCursor();
}

function speedPath(samples, key, duration, scale) {
  const plotWidth = CHART_LAYOUT.width - CHART_LAYOUT.left - CHART_LAYOUT.right;
  const plotHeight = CHART_LAYOUT.bottom - CHART_LAYOUT.top;
  return samples
    .map((sample, index) => {
      const x = CHART_LAYOUT.left + (sample.offset_seconds / duration) * plotWidth;
      const y = CHART_LAYOUT.bottom - (sample[key] / scale.maximum) * plotHeight;
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

/** Round the axis up to a 1/2/5 × 10ⁿ step so tick labels stay readable. */
function chartScale(maximum) {
  const roughStep = maximum / 4;
  const magnitude = 10 ** Math.floor(Math.log10(roughStep));
  const normalized = roughStep / magnitude;
  let normalizedStep = 10;
  if (normalized <= 1.5) normalizedStep = 1;
  else if (normalized <= 3) normalizedStep = 2;
  else if (normalized <= 7) normalizedStep = 5;
  const step = normalizedStep * magnitude;
  const tickCount = Math.max(1, Math.ceil(maximum / step));
  return { maximum: step * tickCount, step, tickCount };
}

function renderChartGrid(duration, scale) {
  elements.chartGrid.replaceChildren();
  const plotHeight = CHART_LAYOUT.bottom - CHART_LAYOUT.top;
  const plotWidth = CHART_LAYOUT.width - CHART_LAYOUT.left - CHART_LAYOUT.right;

  for (let index = 0; index <= scale.tickCount; index += 1) {
    const value = index * scale.step;
    const y = CHART_LAYOUT.bottom - (value / scale.maximum) * plotHeight;
    appendSvgElement("line", {
      class: `chart-grid${index === 0 ? " chart-grid-axis" : ""}`,
      x1: CHART_LAYOUT.left,
      y1: y,
      x2: CHART_LAYOUT.width - CHART_LAYOUT.right,
      y2: y,
    });
    appendSvgElement(
      "text",
      { class: "chart-tick-label", x: CHART_LAYOUT.left - 10, y: y + 4, "text-anchor": "end" },
      formatAxisNumber(value),
    );
  }

  for (let index = 0; index <= X_TICK_COUNT; index += 1) {
    const fraction = index / X_TICK_COUNT;
    const x = CHART_LAYOUT.left + fraction * plotWidth;
    appendSvgElement("line", {
      class: `chart-grid${index === 0 ? " chart-grid-axis" : ""}`,
      x1: x,
      y1: CHART_LAYOUT.top,
      x2: x,
      y2: CHART_LAYOUT.bottom,
    });
    appendSvgElement(
      "text",
      { class: "chart-tick-label", x, y: CHART_LAYOUT.bottom + 23, "text-anchor": "middle" },
      `${(duration * fraction).toFixed(1)} s`,
    );
  }

  appendSvgElement(
    "text",
    {
      class: "chart-axis-title",
      x: -(CHART_LAYOUT.top + CHART_LAYOUT.bottom) / 2,
      y: 13,
      transform: "rotate(-90)",
      "text-anchor": "middle",
    },
    "Speed (BL/s)",
  );
}

function appendSvgElement(name, attributes, text = null) {
  const element = document.createElementNS(SVG_NAMESPACE, name);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
  if (text !== null) element.textContent = text;
  elements.chartGrid.append(element);
}

/** Move the cursor line and the readout to the given media time. */
export function updateChartCursor(mediaTimeValue = elements.video.currentTime) {
  if (state.selectedIndex < 0 || state.chartSamples.length < 2) {
    elements.chartCursor.setAttribute("visibility", "hidden");
    elements.chartHandValue.textContent = "–";
    elements.chartBodyValue.textContent = "–";
    elements.chartFrameLabel.textContent = "Frame –";
    return;
  }
  const move = state.session.moves[state.selectedIndex];
  const offset = mediaTimeValue - mediaTime(move.start_frame);
  const duration = state.chartSamples[state.chartSamples.length - 1].offset_seconds;
  const clampedOffset = Math.max(0, Math.min(offset, duration));
  const x = CHART_LAYOUT.left
    + (clampedOffset / duration) * (CHART_LAYOUT.width - CHART_LAYOUT.left - CHART_LAYOUT.right);
  const sample = nearestChartSample(clampedOffset);
  elements.chartCursor.setAttribute("visibility", "visible");
  elements.chartCursor.setAttribute("x1", x.toFixed(2));
  elements.chartCursor.setAttribute("x2", x.toFixed(2));
  elements.chartHandValue.textContent = formatRelativeSpeed(sample.hand_speed_body_lengths_s);
  elements.chartBodyValue.textContent = formatRelativeSpeed(sample.body_speed_body_lengths_s);
  elements.chartFrameLabel.textContent = `Frame ${sample.frame_idx}`;
}

function nearestChartSample(offset) {
  let low = 0;
  let high = state.chartSamples.length - 1;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (state.chartSamples[middle].offset_seconds < offset) low = middle + 1;
    else high = middle;
  }
  if (low === 0) return state.chartSamples[0];
  const before = state.chartSamples[low - 1];
  const after = state.chartSamples[low];
  return Math.abs(before.offset_seconds - offset) <= Math.abs(after.offset_seconds - offset)
    ? before
    : after;
}
