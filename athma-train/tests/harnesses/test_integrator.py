from __future__ import annotations

from athma_train.harnesses.integrator import (
    CIRCUIT_ID,
    FOM_NAMES,
    SPEC_UNITS,
    TIER,
    measure,
)


GOOD_NETLIST = """
* Inverting Integrator
* Vout/Vin = -1/(s*Rin*Cf) for f >> 16 Hz
* f_unity = 1/(2*pi*10k*10n) = 1.59 kHz

Vdd vdd 0 DC 5.0
Vss vss 0 DC -5.0
Vin Vin 0 AC 1 SIN(0 1 1k)

Rin Vin ninv 10k
Cf ninv Vout 10n
Rf ninv Vout 1Meg

Eamp Vout 0 0 ninv 100k

.ac dec 200 10 100k

.control
  run
  plot vdb(Vout)
.endc

.end
"""

MARGINAL_NETLIST = """
* Integrator with lower unity-gain frequency
Vdd vdd 0 DC 5.0
Vss vss 0 DC -5.0
Vin Vin 0 AC 1 SIN(0 1 1k)

Rin Vin nm 10k
Cf nm Vout 100n
Rf nm Vout 1Meg

Eamp Vout 0 0 nm 100000

.control
  ac dec 100 1 100k
  plot vdb(Vout)
.endc

.end
"""

BROKEN_NETLIST = "this is not spice"


def test_metadata():
    assert CIRCUIT_ID == "integrator"
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
    assert result["dc_gain_db"] is not None


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
