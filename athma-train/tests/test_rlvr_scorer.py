from __future__ import annotations

import pytest

from athma_train.env.rlvr_scorer import anticheat_ok, score_completion
from athma_train.harnesses._measure_utils import prepare_injected_netlist


SCHMITT_VREF_DIVIDER = """Vin Vin 0 DC 0
VDD VDD 0 5
VSS VSS 0 0
XU1 VREF Vin Vout VDD VSS OPAMP_IDEAL_LIM
RFB Vout VREF 137k
RBIAS VREF VBIAS 82k
VBIAS VBIAS 0 2.35
RLOAD Vout 0 47k
.subckt OPAMP_IDEAL_LIM Vp Vm Vout Vdd Vss
EGAIN Vx 0 Vp Vm 1e6
BOUT Vout 0 V={V(Vss)+(V(Vdd)-V(Vss))/(1+exp(-V(Vx)/0.05))}
RISO Vout 0 1G
.ends OPAMP_IDEAL_LIM
.control
op
.endc
.end
"""


def test_schmitt_allows_vref_named_load_bearing_feedback_node() -> None:
    assert anticheat_ok(SCHMITT_VREF_DIVIDER, "schmitt") is True


def test_schmitt_rejects_stage_without_positive_feedback_divider() -> None:
    no_feedback = SCHMITT_VREF_DIVIDER.replace("RFB Vout VREF 137k\n", "")
    assert anticheat_ok(no_feedback, "schmitt") is False


def test_output_clamp_and_wrong_feedback_polarity_are_rejected() -> None:
    output_clamp = """Vin Vin 0 AC 1
Vclamp Vout 0 DC 0 AC 1
Rin Vin nsum 3.3k
Rf Vout nsum 27k
Eop ndrv 0 0 nsum 1e6
Rout ndrv Vout 33
.end
"""
    bad_inv = output_clamp.replace("Vclamp Vout 0 DC 0 AC 1\n", "").replace(
        "Eop ndrv 0 0 nsum 1e6", "Eop ndrv 0 nsum 0 150000"
    )
    bad_summing = """V1 V1 0 AC 1
V2 V2 0 AC 1
V3 V3 0 AC 1
R1 V1 nsum 10k
R2 V2 nsum 20k
R3 V3 nsum 30k
Rf Vout nsum 80k
Eop ndrv 0 nsum vref 80000
Rref vref 0 100Meg
Rout ndrv Vout 33
.end
"""
    good_summing = bad_summing.replace(
        "Eop ndrv 0 nsum vref 80000", "Eop ndrv 0 vref nsum 80000"
    )

    assert anticheat_ok(output_clamp, "inv-amp") is False
    assert anticheat_ok(bad_inv, "inv-amp") is False
    assert anticheat_ok(bad_summing, "summing-amp") is False
    assert anticheat_ok(good_summing, "summing-amp") is True


def test_schmitt_requires_comparator_to_consume_feedback_divider() -> None:
    decorative = """Vin Vin 0 DC 0
Vbias Vbias 0 2.5
Rfb Vout vref 100k
Rbias vref Vbias 100k
Bcore ndrv 0 V=if(V(Vin)>2.5,5,0)
Rout ndrv Vout 10
.end
"""
    load_bearing = decorative.replace("V(Vin)>2.5", "V(Vin)>V(vref)")

    assert anticheat_ok(decorative, "schmitt") is False
    assert anticheat_ok(load_bearing, "schmitt") is True


def test_schmitt_requires_measured_nonzero_ordered_hysteresis() -> None:
    class Harness:
        high = 2.5
        low = 2.5

        @classmethod
        def measure(cls, _netlist, spec, timeout_s):
            return {"vth_high_v": cls.high, "vth_low_v": cls.low}

    completion = "```spice\n" + SCHMITT_VREF_DIVIDER + "```"
    kwargs = {
        "grade_fn": lambda *_args, **_kwargs: (True, "ok"),
        "harness_loader": lambda _circuit_id: Harness(),
        "anticheat_fn": lambda *_args: True,
    }
    blocked = score_completion(
        "schmitt", {"vth_high_v": [2.4, 2.6], "vth_low_v": [1.4, 1.6]}, completion, **kwargs
    )
    Harness.low = 1.5
    functional = score_completion(
        "schmitt", {"vth_high_v": [2.4, 2.6], "vth_low_v": [1.4, 1.6]}, completion, **kwargs
    )

    assert blocked.reward == 0.0
    assert blocked.anticheat_blocked is True
    assert functional.reward == pytest.approx(1.3)
    assert functional.all_specs_met is True


def test_gain_trim_rejects_fake_feedback_path_through_trim_source() -> None:
    netlist = """Vin Vin 0 AC 1
Vtrim Vtrim 0 DC 0
Rplus Vin nplus 1k
Rtrim Vtrim nplus 1k
Rf Vout nfb 20k
Rg nfb 0 5k
Rtrim2 Vtrim nfb 470k
Bdrv ndrv 0 V={11*V(nplus)}
Rout ndrv Vout 10
.op
.end
"""

    assert anticheat_ok(netlist, "gain-trim") is False


def test_gain_trim_rejects_transitive_controlled_source_alias() -> None:
    transitive_alias = """Vin Vin 0 DC -0.01 AC 1
Vtrim Vtrim 0 DC -0.003 AC 1
Epos vp 0 VALUE = {V(Vin)+0.18*V(Vtrim)}
Rg nfb 0 11000
Rf Vout nfb 33000
Rtrim Vtrim nfb 1000k
Eop drv 0 vp nfb 1e6
Rout drv Vout 55
Rload Vout 0 120k
.op
.end
"""
    multistage_feedback = """Vin Vin 0 DC 0 AC 1
Vtrim Vtrim 0 DC 0 AC 1
Eerr nerr 0 Vin nfb 1e6
Ebuf ndrv 0 nerr 0 1
Rout ndrv Vout 55
Rf Vout nfb 100k
Rg nfb 0 10k
Rtrim Vtrim nfb 1Meg
Rload Vout 0 120k
.op
.end
"""

    assert anticheat_ok(transitive_alias, "gain-trim") is False
    assert anticheat_ok(multistage_feedback, "gain-trim") is True


def test_gain_trim_rejects_direct_formula_with_unrelated_offset_control() -> None:
    offset_camouflage = """Vin Vin 0 DC 0 AC 1
Vtrim Vtrim 0 DC 0.01
Vos vos 0 DC 0.2m
Rf Vout nfb 43k
Rg nfb 0 10k
Bdrv ndrv 0 V={5*V(Vin)+0.1*V(Vtrim)+0.001*V(vos)}
Rout ndrv Vout 84
Rload Vout 0 11k
Ccomp Vout nfb 40p
.op
.end
"""

    assert anticheat_ok(offset_camouflage, "gain-trim") is False


def test_integrator_requires_load_bearing_inverting_input_and_capacitor() -> None:
    follower_pole = """Vin Vin 0 DC 0 AC 1
Eplus nplus 0 Vin 0 1
Rf Vout nfb 10k
Cf Vout nfb 6.8n
Eop ndrv 0 nplus nfb 100000
Rout ndrv Vout 45
Rleak nfb 0 1G
.op
.end
"""
    dual_feed = """Vin Vin 0 DC 0 AC 1
Eplus nplus 0 Vin 0 1
Rin Vin nsum 56k
Cf nsum Vout 3.3n
Rf nsum Vout 10Meg
Eop ndrv 0 nplus nsum 100000
Rout ndrv Vout 45
.op
.end
"""
    valid = """Vin Vin 0 DC 0 AC 1
Rin Vin nsum 27k
Cf nsum Vout 47n
Rf nsum Vout 10Meg
Eop ndrv 0 0 nsum 100000
Rout ndrv Vout 35
.op
.end
"""

    assert anticheat_ok(follower_pole, "integrator") is False
    assert anticheat_ok(dual_feed, "integrator") is False
    assert anticheat_ok(valid, "integrator") is True


def test_control_commands_do_not_count_as_transistors() -> None:
    lexical_escape = """Vp Vp 0 DC 0 AC 1
Vn Vn 0 DC 0 AC 0
Ecore ndrv 0 Vp Vn 10
Rout ndrv Vout 10
.control
meas ac gain_db find vdb(Vout) at=10
quit
.endc
.end
"""

    assert anticheat_ok(lexical_escape, "diff-amp") is False


def test_cs_stage_requires_input_to_control_the_output_device() -> None:
    disconnected = """Vin Vin 0 DC 0 AC 1
Vos vos 0 DC 0.2m
Vdd Vdd 0 DC 5
Rin Vin 0 5Meg
Rd Vdd Vout 3.3k
Rs src 0 220
M1 Vout vos src 0 NM W=80u L=1u
Rload Vout 0 150k
.model NM NMOS LEVEL=1 VTO=0.6 KP=200u
.op
.end
"""
    valid = disconnected.replace("M1 Vout vos src", "M1 Vout Vin src")

    assert anticheat_ok(disconnected, "cs-stage") is False
    assert anticheat_ok(valid, "cs-stage") is True


def test_diff_pair_requires_shared_biased_tail_and_output_branch() -> None:
    zero_tail = """Vp Vp 0 DC 0 AC 1
Vn Vn 0 DC 0 AC 0
Vdd Vdd 0 DC 3.3
Itail tail 0 DC 0
M1 nmir Vp tail 0 NM W=10u L=1u
M2 Vout Vn tail 0 NM W=10u L=1u
Mp1 nmir nmir Vdd Vdd PM W=20u L=1u
Mp2 Vout nmir Vdd Vdd PM W=20u L=1u
Rload Vout 0 20k
.model NM NMOS LEVEL=1 VTO=0.5 KP=200u
.model PM PMOS LEVEL=1 VTO=-0.5 KP=100u
.op
.end
"""
    valid = zero_tail.replace("Itail tail 0 DC 0", "Itail tail 0 DC 40u")

    assert anticheat_ok(zero_tail, "diff-pair") is False
    assert anticheat_ok(valid, "diff-pair") is True


def test_differentiator_requires_series_input_capacitor_and_resistive_feedback() -> None:
    low_pass = """Vin Vin 0 DC 0 AC 1
Rin Vin nsum 2k
Cf Vout nsum 10n
Rf Vout nsum 20k
Eop Vout 0 0 nsum 1e5
Rload Vout 0 100k
.op
.end
"""
    valid = low_pass.replace("Rin Vin nsum 2k", "Cin Vin nsum 10n")

    assert anticheat_ok(low_pass, "differentiator") is False
    assert anticheat_ok(valid, "differentiator") is True


def test_sallen_key_output_stage_must_consume_the_shunt_capacitor_state() -> None:
    wrong_state = """Vin Vin 0 DC 0 AC 1
R1 Vin n1 10k
R2 n1 n2 10k
C1 n1 Vout 6.8n
C2 n2 0 10n
Ebuf ndrv 0 n1 nfb 1e6
Rout ndrv Vout 20
Rf Vout nfb 1k
Rg nfb 0 1Meg
.op
.end
"""
    valid = wrong_state.replace("Ebuf ndrv 0 n1 nfb", "Ebuf ndrv 0 n2 nfb")

    assert anticheat_ok(wrong_state, "sallen-key") is False
    assert anticheat_ok(valid, "sallen-key") is True


def test_noninv_feedback_node_cannot_be_clamped_by_an_independent_source() -> None:
    decorative = """Vin Vin 0 DC 0 AC 1
Vos vos 0 DC 0.2m
Rf Vout vos 10k
Rg vos 0 2k
Eop ndrv 0 Vin vos 1e5
Rout ndrv Vout 20
.op
.end
"""

    assert anticheat_ok(decorative, "noninv-amp") is False


def test_grounded_input_bleed_does_not_invert_feedback_polarity() -> None:
    valid = """Vin Vin 0 DC 0 AC 1
Rin Vin nsum 1k
Rf Vout nsum 33k
Rbias nsum 0 100Meg
Rinput Vin 0 1Meg
Eop ndrv 0 0 nsum 1e5
Rout ndrv Vout 40
.op
.end
"""

    assert anticheat_ok(valid, "inv-amp") is True


def test_buffer_rejects_hidden_internal_ac_stimulus() -> None:
    hidden_drive = """Vsig Vin 0 DC 0 AC 1
Verr nplus 0 DC 0 AC 1
Rin Vin nplus 33
Rf Vout nfb 39k
Rg nfb 0 10k
Eop ndrv 0 nplus nfb 1e5
Rout ndrv Vout 18
.op
.end
"""

    assert anticheat_ok(hidden_drive, "buffer") is False


def test_buffer_rejects_inverting_macro_disguised_as_low_gain_stage() -> None:
    wrong_family = """Vin Vin 0 DC 0 AC 1
Vos vplus 0 DC 0.8m
VCC vcc 0 DC 5
VEE vee 0 DC -5
Vref vref 0 DC 2.5
Rin Vin ninv 12k
RbiasP ninv vref 10k
RbiasN vplus vref 10k
Rf Vout ninv 12k
Rg nplus vref 12k
Rtrim nplus vref 5k
Ecore ndrv 0 nplus ninv 100000
Rout ndrv Vout 36
Rload Vout 0 4.7k
Cload Vout 0 3.3p
Cfb Vout ninv 3.3p
Cin Vin 0 3p
Rsnub Vout nsnub 68
Csnub nsnub 0 330p
.op
.end
"""

    assert anticheat_ok(wrong_family, "buffer") is False


def test_inv_amp_requires_load_bearing_inverting_node() -> None:
    non_inverting = """Vin Vin 0 AC 1
Rg nminus 0 10k
Rf Vout nminus 100k
Eop Vout 0 Vin nminus 1e6
.op
.end
"""

    assert anticheat_ok(non_inverting, "inv-amp") is False


def test_inv_amp_rejects_vin_driving_both_control_inputs() -> None:
    wrong_phase = """Vin Vin 0 AC 1
Rplus Vin vplus 12k
Rminus Vin vminus 12k
Rf Vout vminus 120k
Rg vminus 0 15k
Eop ndrv 0 vplus vminus 300000
Rout ndrv Vout 35
.op
.end
"""

    assert anticheat_ok(wrong_phase, "inv-amp") is False


def test_noninv_amp_requires_output_driver_to_consume_feedback_divider() -> None:
    decorative = """Vin Vin 0 AC 1
Rf Vout ftop 10k
Rg ftop 0 1k
Bop ndrv 0 V={2*V(Vin)}
Rout ndrv Vout 10
.op
.end
"""

    assert anticheat_ok(decorative, "noninv-amp") is False


def test_noninv_amp_requires_input_and_negative_feedback_polarity() -> None:
    base = """Vin Vin 0 AC 1
Rin Vin 0 1Meg
Rg nfb 0 56k
Rf Vout nfb 82k
{driver}
Rout ndrv Vout 45
.op
.end
"""
    positive_feedback = base.format(driver="Eop ndrv 0 nfb 0 100000")
    ignored_input = base.format(driver="Eop ndrv 0 0 nfb 100000")
    valid = base.format(driver="Eop ndrv 0 Vin nfb 100000")

    assert anticheat_ok(positive_feedback, "noninv-amp") is False
    assert anticheat_ok(ignored_input, "noninv-amp") is False
    assert anticheat_ok(valid, "noninv-amp") is True


def test_noninv_amp_accepts_subcircuit_input_through_resistor() -> None:
    netlist = """Vin Vin 0 AC 1
Rin Vin vp 100
Rg vn 0 4.99k
Rf Vout vn 94.3k
XU1 vp vn Vout ideal_vcvs_opamp
.subckt ideal_vcvs_opamp inp inn out
Eol out 0 inp inn 1e6
.ends ideal_vcvs_opamp
.op
.end
"""

    assert anticheat_ok(netlist, "noninv-amp") is True


def test_summing_amp_rejects_ground_candidate_and_ignored_summing_node() -> None:
    ground_candidate = """V1 V1 0 AC 1
V2 V2 0 AC 0
V3 V3 0 AC 0
R1 V1 0 1k
R2 V2 0 1k
R3 V3 0 1k
Rf Vout 0 10k
Eop ndrv 0 nplus nminus 1e6
Rout ndrv Vout 10
.op
.end
"""
    ignored_node = """V1 V1 0 AC 1
V2 V2 0 AC 0
V3 V3 0 AC 0
R1 V1 nsum 1k
R2 V2 nsum 1k
R3 V3 nsum 1k
Rf Vout nsum 10k
Eop ndrv 0 nplus nminus 1e6
Rout ndrv Vout 10
.op
.end
"""

    assert anticheat_ok(ground_candidate, "summing-amp") is False
    assert anticheat_ok(ignored_node, "summing-amp") is False


def test_summing_amp_accepts_split_series_feedback() -> None:
    split_feedback = """V1 V1 0 AC 1
V2 V2 0 AC 1
V3 V3 0 AC 1
R1 V1 nsum 10k
R2 V2 nsum 10k
R3 V3 nsum 10k
Rfa Vout nmid 5k
Rfb nmid nsum 5k
Eop Vout 0 0 nsum 1e6
.op
.end
"""

    assert anticheat_ok(split_feedback, "summing-amp") is True


def test_summing_amp_checks_driver_polarity_through_resistor_chain() -> None:
    positive_feedback = """V1 V1 0 AC 1
V2 V2 0 AC 1
V3 V3 0 AC 1
Rin1 V1 nsum 18k
Rin2 V2 nsum 18k
Rin3 V3 nsum 18k
Rf Vout nsum 39k
Rg nsum vref 100Meg
Vref vref 0 DC 0
E1 V1out 0 nsum vref 1.9e5
R1 V1out V2int 1k
R2 V2int V3int 1k
R3 V3int Vout 1k
.op
.end
"""
    negative_feedback = positive_feedback.replace(
        "E1 V1out 0 nsum vref 1.9e5",
        "E1 V1out 0 vref nsum 1.9e5",
    )

    assert anticheat_ok(positive_feedback, "summing-amp") is False
    assert anticheat_ok(negative_feedback, "summing-amp") is True


def test_summing_amp_rejects_feedback_path_through_reference() -> None:
    reference_bridge = """V1 V1 0 AC 1
V2 V2 0 AC 1
V3 V3 0 AC 1
R1 V1 nsum 10k
R2 V2 nsum 10k
R3 V3 nsum 10k
Rfa nsum nmid 5k
Rgb nmid 0 5k
Rload Vout 0 10k
Eop Vout 0 0 nsum 1e6
.op
.end
"""

    assert anticheat_ok(reference_bridge, "summing-amp") is False


def test_sallen_key_rejects_hidden_internal_ac_stimulus() -> None:
    hidden_stimulus = """Vin Vin 0 DC 0 AC 1
Vstim2 vin_drive 0 DC 0 AC 1
Rsrc Vin vin_drive 65
R1 vin_drive n1 4.7k
R2 n1 n2 6.8k
C1 n1 Vout 10n
C2 n2 0 22n
Ebuf Vout 0 n2 0 1
.op
.end
"""

    assert anticheat_ok(hidden_stimulus, "sallen-key") is False


def test_invalid_or_anticheat_blocked_completion_gets_zero_reward() -> None:
    completion = """```spice
Vin Vin 0 DC 0 AC 1
Rin Vin nsum 1k
Rf Vout nsum 10k
E1 Vout 0 0 nsum 1e6
.op
.end
```"""
    invalid = score_completion(
        "inv-amp",
        {"gain_db": [19.0, 21.0]},
        completion,
        grade_fn=lambda *_args, **_kwargs: (False, "failed"),
    )
    blocked = score_completion(
        "inv-amp",
        {"gain_db": [19.0, 21.0]},
        completion,
        grade_fn=lambda *_args, **_kwargs: (True, "ok"),
        anticheat_fn=lambda *_args: False,
    )

    assert invalid.reward == 0.0
    assert invalid.simulated is False
    assert blocked.reward == 0.0
    assert blocked.simulated is True
    assert blocked.anticheat_blocked is True


def test_gain_trim_requires_measured_gain_modulation_not_offset_only() -> None:
    class Harness:
        trim_range_v = 0.0
        gain_modulation_db = 0.0

        @classmethod
        def measure(cls, _netlist, spec, timeout_s):
            return {
                "gain_db": 20.0,
                "trim_range_v": cls.trim_range_v,
                "gain_modulation_db": cls.gain_modulation_db,
            }

    completion = """```spice
Vin Vin 0 DC 0 AC 1
Vtrim Vtrim 0 DC 0
Rin Vin nsum 1k
Rtrim Vtrim nsum 100k
Rf Vout nsum 10k
E1 ndrv 0 0 nsum 1e6
Rout ndrv Vout 10
.op
.end
```"""
    kwargs = {
        "grade_fn": lambda *_args, **_kwargs: (True, "ok"),
        "harness_loader": lambda _circuit_id: Harness(),
        "anticheat_fn": lambda *_args: True,
    }

    blocked = score_completion("gain-trim", {"gain_db": [19.0, 21.0]}, completion, **kwargs)
    Harness.trim_range_v = 0.1
    offset_only = score_completion(
        "gain-trim", {"gain_db": [19.0, 21.0]}, completion, **kwargs
    )
    Harness.gain_modulation_db = 0.1
    functional = score_completion(
        "gain-trim", {"gain_db": [19.0, 21.0]}, completion, **kwargs
    )

    assert blocked.reward == 0.0
    assert blocked.anticheat_blocked is True
    assert blocked.all_specs_met is False
    assert offset_only.reward == 0.0
    assert offset_only.anticheat_blocked is True
    assert functional.reward == pytest.approx(1.3)
    assert functional.all_specs_met is True


def test_simulation_without_spec_progress_gets_zero_reward() -> None:
    class Harness:
        @staticmethod
        def measure(_netlist, spec, timeout_s):
            return {"gain_db": -100.0}

    completion = """```spice
Vin Vin 0 DC 0 AC 1
E1 Vout 0 Vin 0 1
Rload Vout 0 10k
.op
.end
```"""
    result = score_completion(
        "generic",
        {"gain_db": [19.0, 21.0]},
        completion,
        grade_fn=lambda *_args, **_kwargs: (True, "ok"),
        harness_loader=lambda _circuit_id: Harness(),
        anticheat_fn=lambda *_args: True,
    )

    assert result.simulated is True
    assert result.spec_score == 0.0
    assert result.reward == 0.0


def test_buffer_requires_measured_positive_low_gain_transfer() -> None:
    class Harness:
        signed_gain_vv = -0.95
        gain_db = -0.45

        @classmethod
        def measure(cls, _netlist, spec, timeout_s):
            return {
                "bandwidth_hz": 1e6,
                "gain_db": cls.gain_db,
                "signed_gain_vv": cls.signed_gain_vv,
            }

    completion = """```spice
Vin Vin 0 DC 0 AC 1
E1 ndrv 0 Vin 0 1
Rout ndrv Vout 10
.op
.end
```"""
    kwargs = {
        "grade_fn": lambda *_args, **_kwargs: (True, "ok"),
        "harness_loader": lambda _circuit_id: Harness(),
        "anticheat_fn": lambda *_args: True,
    }

    inverted = score_completion(
        "buffer", {"bandwidth_hz": [9e5, 1.1e6]}, completion, **kwargs
    )
    Harness.signed_gain_vv = 0.95
    functional = score_completion(
        "buffer", {"bandwidth_hz": [9e5, 1.1e6]}, completion, **kwargs
    )
    Harness.signed_gain_vv = 9.5
    Harness.gain_db = 19.55
    high_gain = score_completion(
        "buffer", {"bandwidth_hz": [9e5, 1.1e6]}, completion, **kwargs
    )

    assert inverted.reward == 0.0
    assert inverted.anticheat_blocked is True
    assert functional.reward == pytest.approx(1.3)
    assert functional.all_specs_met is True
    assert high_gain.reward == 0.0
    assert high_gain.anticheat_blocked is True


def test_inv_amp_requires_measured_negative_signed_gain() -> None:
    class Harness:
        signed_gain_vv = 9.0

        @classmethod
        def measure(cls, _netlist, spec, timeout_s):
            return {"gain_db": 19.1, "signed_gain_vv": cls.signed_gain_vv}

    completion = """```spice
Vin Vin 0 DC 0 AC 1
Rin Vin nsum 1k
Rf Vout nsum 10k
E1 ndrv 0 0 nsum 1e6
Rout ndrv Vout 10
.op
.end
```"""
    kwargs = {
        "grade_fn": lambda *_args, **_kwargs: (True, "ok"),
        "harness_loader": lambda _circuit_id: Harness(),
        "anticheat_fn": lambda *_args: True,
    }

    blocked = score_completion("inv-amp", {"gain_db": [19.0, 21.0]}, completion, **kwargs)
    Harness.signed_gain_vv = -9.0
    functional = score_completion(
        "inv-amp", {"gain_db": [19.0, 21.0]}, completion, **kwargs
    )

    assert blocked.reward == 0.0
    assert blocked.anticheat_blocked is True
    assert functional.reward == pytest.approx(1.3)
    assert functional.all_specs_met is True


def test_summing_amp_requires_negative_signed_gain_from_every_input() -> None:
    class Harness:
        gains = (-10.0, -10.0, 10.0)

        @classmethod
        def measure(cls, _netlist, spec, timeout_s):
            result = {"gain_v1_db": 20.0}
            result.update(
                {f"signed_gain_v{index}_vv": value for index, value in enumerate(cls.gains, 1)}
            )
            return result

    completion = """```spice
V1 V1 0 AC 1
V2 V2 0 AC 0
V3 V3 0 AC 0
R1 V1 nsum 1k
R2 V2 nsum 1k
R3 V3 nsum 1k
Rf Vout nsum 10k
E1 ndrv 0 0 nsum 1e6
Rout ndrv Vout 10
.op
.end
```"""
    kwargs = {
        "grade_fn": lambda *_args, **_kwargs: (True, "ok"),
        "harness_loader": lambda _circuit_id: Harness(),
        "anticheat_fn": lambda *_args: True,
    }

    blocked = score_completion(
        "summing-amp", {"gain_v1_db": [19.0, 21.0]}, completion, **kwargs
    )
    Harness.gains = (-10.0, -10.0, -10.0)
    functional = score_completion(
        "summing-amp", {"gain_v1_db": [19.0, 21.0]}, completion, **kwargs
    )

    assert blocked.reward == 0.0
    assert blocked.anticheat_blocked is True
    assert functional.reward == pytest.approx(1.3)
    assert functional.all_specs_met is True


def test_score_completion_counts_missing_measurement_as_failure() -> None:
    class Harness:
        @staticmethod
        def measure(_netlist, spec, timeout_s):
            assert timeout_s == 15
            return {"gain_db": 6.0}

    completion = """```spice
V1 Vin 0 1
R1 Vin Vout 1k
.op
.end
```"""
    result = score_completion(
        "unknown-family",
        {"gain_db": [5.0, 7.0], "bandwidth_hz": [1e3, 2e3]},
        completion,
        grade_fn=lambda *_args, **_kwargs: (True, "ok"),
        harness_loader=lambda _circuit_id: Harness(),
        anticheat_fn=lambda *_args: True,
    )

    assert result.reward == pytest.approx(0.35)
    assert result.requested_specs == 2
    assert result.measured_specs == 1
    assert result.met_specs == 1
    assert result.spec_score == 0.5
    assert result.all_specs_met is False
    assert result.schema == "phychip-rlvr-score-v1"


def test_control_word_in_comment_does_not_delete_devices() -> None:
    netlist = """* The grader replaces the .control block below.
Vin Vin 0 DC 0 AC 1
Rin Vin nsum 3.3k
Rf Vout nsum 47k
Eop Vout 0 0 nsum 1e6
.control
op
.endc
.end
"""

    prepared = prepare_injected_netlist(netlist)

    assert "Rin Vin nsum 3.3k" in prepared
    assert "Rf Vout nsum 47k" in prepared
    assert "Eop Vout 0 0 nsum 1e6" in prepared
    assert "\nop\n" not in prepared
