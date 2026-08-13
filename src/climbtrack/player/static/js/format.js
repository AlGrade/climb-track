/** Number and time formatting shared by the views. */

/** `mm:ss.mmm` — the frame-exact readout used in the editor and the move list. */
export function formatTime(seconds) {
  if (!Number.isFinite(seconds)) return "00:00.000";
  const minutes = Math.floor(seconds / 60);
  const remaining = seconds - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remaining.toFixed(3).padStart(6, "0")}`;
}

/** `mm:ss` — the coarser readout next to the transport controls. */
export function formatControlTime(seconds) {
  if (!Number.isFinite(seconds)) return "00:00";
  const minutes = Math.floor(seconds / 60);
  const remaining = Math.floor(seconds - minutes * 60);
  return `${String(minutes).padStart(2, "0")}:${String(remaining).padStart(2, "0")}`;
}

export function formatRelativeSpeed(value) {
  return `${value.toFixed(2)} BL/s`;
}

/** Signed body lengths, using a true minus sign rather than a hyphen. */
export function formatSignedBodyLengths(value) {
  return `${value < 0 ? "−" : "+"}${Math.abs(value).toFixed(2)} BL`;
}

/** Trim trailing zeroes so chart ticks read 0.5 rather than 0.50. */
export function formatAxisNumber(value) {
  return Number(value.toFixed(2)).toString();
}
