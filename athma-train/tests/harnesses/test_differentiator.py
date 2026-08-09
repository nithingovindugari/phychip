from __future__ import annotations

from athma_train.harnesses.differentiator import (
    CIRCUIT_ID,
    FOM_NAMES,
    SPEC_UNITS,
    TIER,
    measure,
)


GOOD_NETLIST = """
* Op-amp Differentiator
* Vout = -Rf*C1*dVin/dt, with high-frequency gain capped by Rin.

Vdd vdd 0 DC 5
Vss vss 0 DC -5
Vin Vin 0 AC 1 SIN(0 1 1k)

Rin Vin nm 1k
C1 nm ninv 10n
Rf Vout ninv 10k

Eamp Vout 0 0 ninv 100k
Rload Vout 0 100k

.control
  ac dec 100 10 10Meg
  tran 1u 5m
  plot v(Vin) v(Vout)
.endc

.end
"""

MARGINAL_NETLIST = """
* Differentiator with lower time constant than the good reference.

Vdd vdd 0 DC 5
Vss vss 0 DC -5
Vin Vin 0 AC 1 SIN(0 0.2 1k)

Rin Vin nm 2k
C1 nm ninv 1n
Rf Vout ninv 20k

Eamp Vout 0 0 ninv 100k
Rload Vout 0 100k

.control
  ac dec 100 10 10Meg
  tran 1u 5m
  plot v(Vout)
.endc

.end
"""

BROKEN_NETLIST = "this is not spice"


def test_metadata():
    assert CIRCUIT_ID == "differentiator"
    assert TIER in (0, 1, 2)
    assert len(FOM_NAMES) >= 3
    for fom in FOM_NAMES:
        assert fom in SPEC_UNITS


def test_good_netlist_extracts_all_foms():
    result = measure(GOOD_NETLIST)
    assert set(result.keys()) == set(FOM_NAMES)
    non_none = [value for value in result.values() if value is not None]
    assert len(non_none) >= 2


def test_marginal_netlist_extracts_frequency_foms():
    result = measure(MARGINAL_NETLIST)
    assert set(result.keys()) == set(FOM_NAMES)
    assert result["unity_gain_freq_hz"] is not None
    assert result["hf_gain_db"] is not None


def test_broken_netlist_returns_all_none():
    result = measure(BROKEN_NETLIST)
    assert set(result.keys()) == set(FOM_NAMES)
    assert all(value is None for value in result.values())


def test_idempotent():
    first = measure(GOOD_NETLIST)
    second = measure(GOOD_NETLIST)
    for key in FOM_NAMES:
        if first[key] is None and second[key] is None:
            continue
        assert abs((first[key] or 0) - (second[key] or 0)) < 1e-9, key
