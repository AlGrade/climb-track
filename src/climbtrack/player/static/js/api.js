/** The two endpoints the player talks to. */

export async function fetchSession() {
  const response = await fetch("/api/session");
  if (!response.ok) throw new Error(`Could not load player data (${response.status})`);
  return response.json();
}

/**
 * Replace the whole move list.
 *
 * `expectedRevision` is what makes concurrent edits safe: the server rejects the
 * write when another tab saved in the meantime, rather than silently discarding
 * that tab's work.
 */
export async function saveMoves(expectedRevision, moves) {
  const response = await fetch("/api/moves", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ expected_revision: expectedRevision, moves }),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Save failed (${response.status})`);
  return payload;
}
