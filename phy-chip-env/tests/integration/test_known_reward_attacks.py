from __future__ import annotations

import json
from pathlib import Path

import pytest

from phy_chip_env import score

TARGETS = {
    "inv-amp": {
        "bandwidth_hz": [314825.29326631484, 584675.5446374419],
        "gain_db": [13.9993, 25.9987],
        "input_offset_v": [8.83526e-30, 1.640834e-29],
        "output_swing_v": [27.99692, 51.99428],
    },
    "noninv-amp": {
        "bandwidth_hz": [1396455.769092665, 2593417.856886378],
        "gain_db": [13.99937, 25.99883],
        "input_offset_v": [2.76101e-31, 5.12759e-31],
        "output_swing_v": [41.9958, 77.9922],
    },
    "cs-stage": {
        "bandwidth_hz": [3436788.875310788, 6382607.911291463],
        "gain_db": [14.21434, 26.39806],
        "output_swing_v": [1.4502306, 2.6932854],
        "quiescent_current_a": [0.00033641755, 0.00062477545],
    },
    "current-mirror": {"current_ratio": [0.0, 0.0]},
    "sallen-key": {
        "cutoff_hz": [49154.76106946035, 91287.41341471208],
        "dc_gain_db": [4.21442, 7.82678],
        "q_factor": [0.49415728750968235, 0.917720676803696],
        "stopband_atten_db": [4.944415, 9.182485],
    },
    "ldo": {
        "line_reg_v_per_v": [0.0, 0.0],
        "load_reg_ohm": [0.042, 0.078],
        "vout_dc_v": [1.259622, 2.339298],
    },
    "buffer": {"bandwidth_hz": [3724434.064420321, 6916806.119637738]},
    "gain-trim": {"gain_db": [12.0868, 18.0868]},
    "integrator": {"unity_gain_freq_hz": [100.0, 200.0]},
    "diff-amp": {"differential_gain_db": [20.0, 30.0]},
    "differentiator": {"unity_gain_freq_hz": [1000.0, 2000.0]},
}


@pytest.mark.integration
def test_all_22_known_attacks_stay_below_success() -> None:
    path = Path(__file__).parents[1] / "fixtures" / "known_reward_attacks.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    results = []
    for row in rows:
        completion = f"```spice\n{row['netlist']}\n```"
        result = score(row["circuit_id"], TARGETS[row["circuit_id"]], completion)
        results.append((row["hack_type"], result))

    assert len(results) == 22
    assert all(not result.all_specs_met for _, result in results)
    assert all(result.reward < 1.0 for _, result in results)
