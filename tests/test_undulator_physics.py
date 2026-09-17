"""The slow tests: real undulator emission.

Marked slow because each one costs tens of seconds. They exist because the
geometric-source tests cannot catch a mistake in the source, and the source is
where the absolute flux and power numbers come from — the two numbers a
beamline scientist will check first.

Run them with:  pytest -m slow
"""

from __future__ import annotations

import pytest

from xrt_mcp import templates
from xrt_mcp.spec import K_for_energy, Undulator, resonance_energy_eV
from xrt_mcp.tracing import trace

pytestmark = pytest.mark.slow


def test_tuning_is_self_consistent():
    """Solving for K and then asking where the harmonic sits must round-trip."""
    for energy in (7000.0, 9000.0, 14000.0):
        k = K_for_energy(energy, 21.0, 3.0, 5)
        assert resonance_energy_eV(21.0, k, 3.0, 5) == pytest.approx(energy, rel=1e-9)


def test_an_unreachable_energy_says_so():
    with pytest.raises(ValueError, match="cannot reach"):
        Undulator(tune_to_eV=9000.0, harmonic=1)


def test_undulator_gives_absolute_flux_and_power():
    spec = templates.get("nsls2-hard-xray", energy_eV=9000.0)
    result = trace(spec, nrays=8000)

    assert result.source_flux_ph_s is not None
    assert result.source_power_W is not None

    sample = result.by_name("sample")
    front = result.by_name("front_end")

    # flux can only fall along a beamline
    assert sample.flux_ph_s <= front.flux_ph_s <= result.source_flux_ph_s
    # and a monochromated 9 keV beam at a real ring is not 1e3 or 1e25 ph/s
    assert 1e8 < sample.flux_ph_s < 1e16
    assert 8990 < sample.mean_energy_eV < 9010


def test_the_monochromator_agrees_with_the_crystal_maths():
    """Two independent paths to the same number: the Darwin width computed from
    the structure factor, and the bandwidth that actually survives a trace."""
    from xrt_mcp.reference import crystal_at_energy

    predicted = crystal_at_energy("Si111", 9000.0)["energy_bandwidth_eV"]
    result = trace(templates.get("nsls2-hard-xray", energy_eV=9000.0), nrays=8000)
    measured = result.by_name("sample").energy_fwhm_eV

    # the trace is always the wider of the two: the incident beam has divergence,
    # and divergence maps onto energy through cot(theta_B)
    assert predicted <= measured < 6 * predicted
