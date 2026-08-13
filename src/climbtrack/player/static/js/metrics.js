/** The metric tiles below the video. */

import { renderSpeedChart } from "./chart.js";
import { elements } from "./dom.js";
import { formatRelativeSpeed, formatSignedBodyLengths } from "./format.js";
import { state } from "./state.js";

export function renderMetrics() {
  if (state.selectedIndex < 0 || !state.session.moves.length) {
    state.chartSamples = [];
    elements.metricsCard.hidden = true;
    return;
  }
  const move = state.session.moves[state.selectedIndex];
  const metrics = state.metrics.find((row) => row.move_id === move.move_id);
  elements.metricsCard.hidden = false;
  if (!metrics) {
    // Boundaries were edited after the metrics were computed. Showing the stale
    // numbers would be worse than showing none.
    state.chartSamples = [];
    elements.metricsGrid.hidden = true;
    elements.speedChartWrap.hidden = true;
    elements.metricsNote.textContent = "Restart the player to recalculate.";
    return;
  }
  elements.metricsGrid.hidden = false;
  elements.speedChartWrap.hidden = false;
  elements.handMaxSpeed.textContent = formatRelativeSpeed(metrics.hand_max_speed_body_lengths_s);
  elements.handMaxSpeedPx.textContent = `${metrics.hand_max_speed_px_s.toFixed(0)} px/s`;
  elements.bodyMaxSpeed.textContent = formatRelativeSpeed(metrics.body_max_speed_body_lengths_s);
  elements.bodyMaxSpeedPx.textContent = `${metrics.body_max_speed_px_s.toFixed(0)} px/s`;
  elements.handMeanSpeed.textContent = formatRelativeSpeed(metrics.hand_mean_speed_body_lengths_s);
  elements.handPath.textContent = `Path ${metrics.hand_path_length_px.toFixed(0)} px`;
  elements.bodyMeanSpeed.textContent = formatRelativeSpeed(metrics.body_mean_speed_body_lengths_s);
  elements.bodyPath.textContent = `Path ${metrics.body_path_length_px.toFixed(0)} px`;
  renderPostureMetrics(metrics);
  elements.metricsNote.textContent = "BL/s = body lengths per second";
  renderSpeedChart(move);
}

/**
 * Posture is read where the moving hand comes to rest, which is the grasp on a
 * completed move and the bottom of the fall on a failed one. That frame is
 * deliberately earlier than the move end, because a move only closes once the
 * body and legs settle as well.
 */
function renderPostureMetrics(metrics) {
  elements.hipRise.textContent = formatSignedBodyLengths(metrics.hip_rise_body_lengths);
  elements.hipRiseNote.textContent = "until the hand rests";
  elements.hipBelowHand.textContent = `${metrics.hip_below_hand_body_lengths.toFixed(2)} BL`;
  elements.hipBelowHandNote.textContent = "when the hand rests";
  elements.handSettle.textContent = `${metrics.hand_settle_offset_seconds.toFixed(2)} s`;
  elements.handSettleNote.textContent = "after release";

  const lag = metrics.coordination_lag_seconds;
  const correlation = metrics.coordination_correlation;
  if (lag === null || correlation === null) {
    // The server sends null when the correlation peak sits on the edge of the
    // searched range or one curve is flat. Showing a number would invent one.
    elements.torsoLead.textContent = "–";
    elements.torsoLeadNote.textContent = "undefined";
    return;
  }
  const milliseconds = Math.abs(lag * 1000).toFixed(0);
  elements.torsoLead.textContent = `${lag < 0 ? "−" : "+"}${milliseconds} ms`;
  elements.torsoLeadNote.textContent = `r = ${correlation.toFixed(2)}`;
}
