from __future__ import annotations

from phy_chip_env import get_harness, list_harnesses


def test_registry_contains_exactly_23_validated_harnesses() -> None:
    infos = list_harnesses()

    assert len(infos) == 23
    assert len({info.circuit_id for info in infos}) == 23
    assert {info.tier for info in infos} == {0, 1, 2}
    for info in infos:
        module = get_harness(info.circuit_id)
        assert module is not None
        assert module.CIRCUIT_ID == info.circuit_id
        assert set(info.fom_names) == set(info.spec_units)


def test_unknown_harness_fails_closed() -> None:
    assert get_harness("not-a-circuit") is None
