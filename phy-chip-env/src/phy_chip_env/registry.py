"""Explicit, validated registry for the 23 v0.1 circuit harnesses."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Protocol


class Harness(Protocol):
    CIRCUIT_ID: str
    TIER: int
    FOM_NAMES: tuple[str, ...]
    SPEC_UNITS: dict[str, str]

    def measure(
        self,
        netlist: str,
        spec: dict | None = None,
        timeout_s: int = 30,
    ) -> dict[str, float | None]: ...


@dataclass(frozen=True)
class HarnessInfo:
    circuit_id: str
    module_name: str
    tier: int
    fom_names: tuple[str, ...]
    spec_units: dict[str, str]


_MODULE_BY_CIRCUIT_ID = {
    "bandgap": "bandgap",
    "boost": "boost",
    "bridge-ina": "bridge_ina",
    "buck": "buck",
    "buffer": "buffer",
    "cs-stage": "cs_stage",
    "current-mirror": "current_mirror",
    "current-sense": "current_sense",
    "diff-amp": "diff_amp",
    "diff-pair": "diff_pair",
    "differentiator": "differentiator",
    "gain-trim": "gain_trim",
    "h-bridge": "h_bridge",
    "ina": "ina",
    "integrator": "integrator",
    "inv-amp": "inv_amp",
    "ldo": "ldo",
    "level-shift": "level_shift",
    "noninv-amp": "noninv_amp",
    "sallen-key": "sallen_key",
    "schmitt": "schmitt",
    "summing-amp": "summing_amp",
    "tia": "tia",
}
_CACHE: dict[str, ModuleType] = {}


def _validate_harness(circuit_id: str, module_name: str, module: ModuleType) -> ModuleType:
    required = ("CIRCUIT_ID", "TIER", "FOM_NAMES", "SPEC_UNITS", "measure")
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"harness {circuit_id!r} is missing {missing}")
    if module.CIRCUIT_ID != circuit_id:
        raise RuntimeError(
            f"harness registry mismatch: {module_name} declares {module.CIRCUIT_ID!r}, "
            f"expected {circuit_id!r}"
        )
    if not isinstance(module.FOM_NAMES, tuple) or not module.FOM_NAMES:
        raise RuntimeError(f"harness {circuit_id!r} must declare non-empty FOM_NAMES")
    if set(module.FOM_NAMES) != set(module.SPEC_UNITS):
        raise RuntimeError(f"harness {circuit_id!r} FOM_NAMES/SPEC_UNITS disagree")
    if not callable(module.measure):
        raise RuntimeError(f"harness {circuit_id!r} measure is not callable")
    return module


def get_harness(circuit_id: str) -> ModuleType | None:
    """Return the validated harness for a canonical circuit-family identifier."""

    if circuit_id not in _MODULE_BY_CIRCUIT_ID:
        return None
    if circuit_id not in _CACHE:
        module_name = _MODULE_BY_CIRCUIT_ID[circuit_id]
        module = importlib.import_module(f"phy_chip_env.harnesses.{module_name}")
        _CACHE[circuit_id] = _validate_harness(circuit_id, module_name, module)
    return _CACHE[circuit_id]


def list_harnesses() -> tuple[HarnessInfo, ...]:
    """Return stable metadata for every packaged harness."""

    infos: list[HarnessInfo] = []
    for circuit_id, module_name in _MODULE_BY_CIRCUIT_ID.items():
        module = get_harness(circuit_id)
        assert module is not None
        infos.append(
            HarnessInfo(
                circuit_id=circuit_id,
                module_name=module_name,
                tier=int(module.TIER),
                fom_names=tuple(module.FOM_NAMES),
                spec_units=dict(module.SPEC_UNITS),
            )
        )
    return tuple(infos)
