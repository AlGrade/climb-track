/** Indirection that lets controllers ask for a full re-render.
 *
 * The full render is composed in main.js out of the individual view modules. If
 * controllers imported that composition directly, every view would end up
 * importing its callers back. Registering the renderer once at start-up keeps the
 * dependencies pointing one way.
 */

let renderer = () => {};

export function setRenderer(callback) {
  renderer = callback;
}

export function requestRender() {
  renderer();
}
