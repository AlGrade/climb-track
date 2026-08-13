"""Static checks on the player front end.

There is no browser in the test suite, so these guard the mistakes that would
otherwise only show up as a blank page: a stylesheet or module that the markup
references but that does not exist, an import that points at a missing file or a
name nobody exports, and an import cycle.
"""

import re
from pathlib import Path

import pytest

from climbtrack.player import server

STATIC_ROOT = Path(server.__file__).with_name("static")
JS_ROOT = STATIC_ROOT / "js"

ASSET_REFERENCE = re.compile(r'(?:href|src)="/assets/([^"]+)"')
IMPORT_STATEMENT = re.compile(r'import\s*\{([^}]*)\}\s*from\s*["\']\./([^"\']+)["\']')
NAMED_EXPORT = re.compile(r"^export\s+(?:async\s+)?(?:function|const|let|class)\s+([\w$]+)", re.M)


def js_modules() -> list[Path]:
    return sorted(JS_ROOT.glob("*.js"))


def exported_names(module: Path) -> set[str]:
    return set(NAMED_EXPORT.findall(module.read_text(encoding="utf-8")))


def imports_of(module: Path) -> list[tuple[list[str], str]]:
    text = module.read_text(encoding="utf-8")
    return [
        ([name.strip() for name in names.split(",") if name.strip()], target)
        for names, target in IMPORT_STATEMENT.findall(text)
    ]


def test_markup_references_only_existing_assets() -> None:
    markup = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    referenced = ASSET_REFERENCE.findall(markup)

    assert referenced, "index.html should reference at least one asset"
    for relative in referenced:
        resolved = server.resolve_asset(STATIC_ROOT, relative)
        assert resolved is not None and resolved.is_file(), f"missing asset: {relative}"
        assert resolved.suffix in server.ASSET_CONTENT_TYPES, f"unservable asset: {relative}"


@pytest.mark.parametrize("module", js_modules(), ids=lambda path: path.name)
def test_module_imports_resolve(module: Path) -> None:
    for names, target in imports_of(module):
        target_path = JS_ROOT / target
        assert target_path.is_file(), f"{module.name} imports missing ./{target}"
        available = exported_names(target_path)
        missing = sorted(set(names) - available)
        assert not missing, f"{module.name}: ./{target} does not export {missing}"


def test_module_graph_has_no_cycles() -> None:
    graph = {module.name: [target for _, target in imports_of(module)] for module in js_modules()}
    visiting: set[str] = set()
    settled: set[str] = set()
    cycles: list[str] = []

    def walk(name: str, stack: list[str]) -> None:
        if name in visiting:
            cycles.append(" -> ".join([*stack[stack.index(name) :], name]))
            return
        if name in settled:
            return
        visiting.add(name)
        for dependency in graph.get(name, []):
            walk(dependency, [*stack, name])
        visiting.discard(name)
        settled.add(name)

    for name in graph:
        walk(name, [])

    assert not cycles, f"import cycles: {sorted(set(cycles))}"


def test_every_module_is_reachable_from_the_entry_point() -> None:
    graph = {module.name: [target for _, target in imports_of(module)] for module in js_modules()}
    reached = {"main.js"}
    queue = ["main.js"]
    while queue:
        for dependency in graph.get(queue.pop(), []):
            if dependency not in reached:
                reached.add(dependency)
                queue.append(dependency)

    assert set(graph) == reached, f"unreachable modules: {sorted(set(graph) - reached)}"
