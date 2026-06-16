from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence

from athma_train.spice_gate import has_device_line, normalize_netlist


_INJECTED_SKIP_DIRECTIVES = (
    ".op",
    ".tran",
    ".dc",
    ".ac",
    ".noise",
    ".disto",
    ".measure",
    ".meas",
    ".print",
    ".plot",
    ".four",
)


def prepare_injected_netlist(user_netlist: str) -> str:
    nl = normalize_netlist(user_netlist)
    nl = re.sub(r"\.control\b.*?\.endc\b", "", nl, flags=re.DOTALL | re.IGNORECASE)
    nl = re.sub(r"\.end\s*$", "", nl, flags=re.IGNORECASE).rstrip()
    kept: list[str] = []
    for line in nl.splitlines():
        stripped = line.lstrip().lower()
        if any(stripped.startswith(directive) for directive in _INJECTED_SKIP_DIRECTIVES):
            continue
        kept.append(line)
    cleaned = "\n".join(kept)
    return re.sub(r"-\s*\{([^}]+)\}", r"{-\1}", cleaned).rstrip()


def has_prepared_device_line(user_netlist: str) -> bool:
    return has_device_line(prepare_injected_netlist(user_netlist))


def netlist_nodes(nl: str) -> set[str]:
    nodes: set[str] = set()
    for line in nl.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("*", ".")):
            continue
        parts = stripped.split()
        if parts[0].lower().startswith(("r", "c", "l", "v", "i")) and len(parts) >= 3:
            nodes.update((parts[1].lower(), parts[2].lower()))
        elif parts[0].lower().startswith(("m", "q")) and len(parts) >= 5:
            nodes.update(part.lower() for part in parts[1:5])
        elif parts[0].lower().startswith(("e", "g")) and len(parts) >= 5:
            nodes.update(part.lower() for part in parts[1:5])
        elif parts[0].lower().startswith("x") and len(parts) >= 3:
            nodes.update(part.lower() for part in parts[1:-1])
    return nodes


def pick_node(nl: str, candidates: Sequence[str], fallback: str) -> str:
    nodes = netlist_nodes(nl)
    for candidate in candidates:
        if candidate.lower() in nodes:
            return candidate
    return fallback


def parse_numeric_rows(stdout: str, columns: int = 2) -> list[tuple[float, ...]]:
    rows: list[tuple[float, ...]] = []
    pattern = re.compile(
        r"^\s*\d+\s+"
        + r"\s+".join([r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"] * columns)
        + r"\s*$",
        re.MULTILINE,
    )
    for match in pattern.finditer(stdout):
        try:
            values = tuple(float(value) for value in match.groups())
        except ValueError:
            continue
        if all(math.isfinite(value) for value in values):
            rows.append(values)
    return rows


def _section(stdout: str, marker: str | None) -> str:
    if marker is None:
        return stdout
    start = stdout.find(marker)
    if start < 0:
        return stdout
    tail = stdout[start + len(marker) :]
    next_marker = tail.find("__")
    return tail if next_marker < 0 else tail[:next_marker]


def parse_ac_points(
    stdout: str,
    value_column: int = 1,
    marker: str | None = None,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in parse_numeric_rows(_section(stdout, marker), max(value_column + 1, 2)):
        frequency = row[0]
        value = row[value_column]
        if frequency > 0.0:
            points.append((frequency, value))
    return points


def low_frequency_value(
    points: Sequence[tuple[float, float]],
    *,
    stop_hz: float = 1.0e3,
    reducer: Callable[[list[float]], float] = max,
) -> float | None:
    values = [value for frequency, value in points if frequency <= stop_hz]
    if not values and points:
        values = [points[0][1]]
    return reducer(values) if values else None


def interpolate_crossing_hz(
    points: Sequence[tuple[float, float]],
    target: float,
    *,
    direction: str = "fall",
) -> float | None:
    for (f0, y0), (f1, y1) in zip(points, points[1:]):
        if f0 <= 0.0 or f1 <= 0.0 or y0 == y1:
            continue
        if direction == "fall":
            crossed = y0 >= target >= y1
        elif direction == "rise":
            crossed = y0 <= target <= y1
        else:
            crossed = (y0 - target) * (y1 - target) <= 0.0
        if not crossed:
            continue
        ratio = (target - y0) / (y1 - y0)
        log_f = math.log10(f0) + ratio * (math.log10(f1) - math.log10(f0))
        return 10.0**log_f
    return None


def three_db_frequency_hz(
    stdout: str,
    *,
    value_column: int = 1,
    low_stop_hz: float = 1.0e3,
    is_db: bool = True,
    fallback_hz: float | None = None,
    marker: str | None = None,
) -> float | None:
    points = parse_ac_points(stdout, value_column=value_column, marker=marker)
    low_value = low_frequency_value(points, stop_hz=low_stop_hz)
    if low_value is None:
        return None
    target = low_value - 3.0 if is_db else low_value / math.sqrt(2.0)
    return interpolate_crossing_hz(points, target, direction="fall") or fallback_hz
