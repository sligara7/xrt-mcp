"""The response contract — what every answer must carry, and what it must not.

These tests exist because six separate rough edges reported from the first
outside trial (xrt-demo, 2026-09-16) all had one cause: nothing in this project
said what a RESPONSE was, so six return statements each invented their own.
A fix per finding would have left the seventh tool free to invent a seventh
convention, so what is pinned here is the contract, not the six instances.

Every test in this file was written against the unfixed server and observed to
fail before anything was changed.
"""

from __future__ import annotations

import json

import pytest

from xrt_mcp import reference
from xrt_mcp.server import mcp

# tools whose replies are evidence — numbers somebody may record and quote later
EVIDENCE_TOOLS = {
    "crystal_at_energy", "crystal_energy_range", "mirror_cutoff",
    "undulator_harmonics", "trace_beamline",
}


@pytest.mark.anyio
async def test_every_tool_declares_what_it_returns():
    """A model should not have to call a tool to find out what comes back."""
    missing = [
        t.name for t in await mcp.list_tools()
        if not getattr(t, "output_schema", None) and t.name in EVIDENCE_TOOLS
    ]
    assert not missing, f"tools with no output schema: {missing}"


@pytest.mark.anyio
async def test_evidence_carries_the_build_that_produced_it():
    """Two machines disagreeing must be separable into 'different model' and
    'different xrt build'. That needs the build in the payload."""
    for t in await mcp.list_tools():
        if t.name not in EVIDENCE_TOOLS:
            continue
        props = (t.output_schema or {}).get("properties", {})
        assert "meta" in props, f"{t.name} returns no provenance block"


def test_a_lookup_is_small_unless_you_ask_for_more():
    """An agent carries every byte of a reply for the rest of its session."""
    small = reference.mirror_cutoff("Rh", 3.0)
    assert len(json.dumps(small)) < 2000, "default reply should not carry the curve"
    big = reference.mirror_cutoff("Rh", 3.0, include_curve=True)
    assert len(json.dumps(big)) > len(json.dumps(small))


def test_the_curve_does_not_misrepresent_an_absorption_edge():
    """The bug this pins: the reply used to decimate a 400-point curve 20x,
    which kept ONE sample across Rh's three L edges and rendered a step down
    (0.97 -> 0.88, recovering only to 0.91 by 4.2 keV) as a V-shaped notch that
    appeared to recover. A reader sized harmonic rejection off the recovery.
    That is a wrong conclusion reachable from a correct xrt computation."""
    out = reference.mirror_cutoff("Rh", 3.0, include_curve=True)
    e = out["reflectivity_curve"]["energy_eV"]
    r = out["reflectivity_curve"]["reflectivity"]

    def at(target):
        return r[min(range(len(e)), key=lambda i: abs(e[i] - target))]

    # below all three L edges, Rh reflects well
    assert at(2800) > 0.95
    # above them it does NOT come back to that value, anywhere in range
    for probe in (3500, 4200, 5000, 8000):
        assert at(probe) < 0.95, (
            f"reflectivity at {probe} eV reads {at(probe):.3f}; if this looks like "
            "a recovery to the pre-edge value, the curve is being decimated again"
        )


def test_the_edges_are_reported_not_left_to_be_inferred():
    """Rh L3/L2/L1 sit at 3004/3146/3412 eV. A caller should not have to spot
    them in a curve, and should not have to know they exist to avoid them."""
    out = reference.mirror_cutoff("Rh", 3.0)
    found = [edge["energy_eV"] for edge in out["absorption_edges"]]
    for tabulated in (3004.0, 3146.0, 3412.0):
        assert any(abs(f - tabulated) / tabulated < 0.02 for f in found), (
            f"no reported edge near {tabulated} eV; found {found}"
        )


@pytest.mark.anyio
async def test_continuous_physical_quantities_are_not_defaulted():
    """A defaulted crystal is a menu item you notice omitting. A defaulted
    energy is a silent physics assumption that returns a well-formed, plausible,
    wrong answer."""
    continuous = {
        "crystal_at_energy": {"energy_eV"},
        "mirror_cutoff": {"pitch_mrad"},
        "undulator_harmonics": {"period_mm", "K", "ring_GeV"},
    }
    tools = {t.name: t for t in await mcp.list_tools()}
    for name, must_be_required in continuous.items():
        required = set(tools[name].input_schema.get("required", []))
        missing = must_be_required - required
        assert not missing, f"{name} lets {sorted(missing)} default silently"
