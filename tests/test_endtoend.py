"""An MCP client talks to the server over stdio and checks the answers.

This is the test that matters: it exercises the real protocol, not the Python
functions underneath it, so a tool whose signature the SDK cannot serialise
fails here rather than in front of somebody.

Everything traced here uses a geometric source, which takes under a second.
The undulator path is covered by `test_undulator_physics`, which is slow and
marked as such.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


def _content_text(result) -> str:
    return "\n".join(c.text for c in result.content if getattr(c, "type", None) == "text")


def _images(result) -> list:
    return [c for c in result.content if getattr(c, "type", None) == "image"]


async def _session():
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "xrt_mcp"], cwd=str(ROOT)
    )
    return stdio_client(params)


@pytest.mark.anyio
async def test_tools_are_listed():
    async with await _session() as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = {t.name for t in (await session.list_tools()).tools}
    assert {
        "list_beamlines", "load_beamline", "trace_beamline", "beam_image",
        "scan_element", "crystal_at_energy", "mirror_cutoff",
    } <= tools


@pytest.mark.anyio
async def test_crystal_lookup_matches_textbook():
    """Si(111) at 9 keV: Bragg angle 12.7 deg, intrinsic bandwidth ~1.2 eV."""
    async with await _session() as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "crystal_at_energy", {"crystal": "Si111", "energy_eV": 9000}
            )
    data = json.loads(_content_text(result))
    assert 12.5 < data["bragg_angle_deg"] < 12.9
    assert 1.0 < data["energy_bandwidth_eV"] < 1.5


@pytest.mark.anyio
async def test_trace_focuses_and_monochromates():
    """The thing the demo claims: a DCM cuts a 50 eV band to about 1.5 eV, and
    a toroidal mirror puts a millimetre-wide beam into tens of microns."""
    async with await _session() as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            loaded = await session.call_tool(
                "load_beamline", {"name": "nsls2-hard-xray-fast", "energy_eV": 9000}
            )
            spec = json.loads(_content_text(loaded))
            traced = await session.call_tool(
                "trace_beamline", {"spec": spec, "nrays": 20000}
            )
    data = json.loads(_content_text(traced))
    screens = {s["name"]: s for s in data["screens"]}

    front, sample = screens["front_end"], screens["sample"]
    assert front["energy_fwhm_eV"] > 40          # white beam, the sampled window
    assert sample["energy_fwhm_eV"] < 4          # monochromated
    assert front["x_fwhm_um"] > 500              # a millimetre-scale beam
    assert sample["x_fwhm_um"] < 80              # focused
    assert sample["z_fwhm_um"] < 20
    assert 8990 < sample["mean_energy_eV"] < 9010
    assert sample["flux_ph_s"] is None           # geometric source: no flux claim


@pytest.mark.anyio
async def test_beam_image_comes_back_as_an_image():
    async with await _session() as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            loaded = await session.call_tool(
                "load_beamline", {"name": "nsls2-hard-xray-fast"}
            )
            spec = json.loads(_content_text(loaded))
            result = await session.call_tool(
                "beam_image", {"spec": spec, "screen": "sample", "nrays": 5000}
            )
    images = _images(result)
    assert len(images) == 1
    assert images[0].mime_type == "image/png"
    assert len(images[0].data) > 2000
    assert "sample at 45 m" in _content_text(result)


@pytest.mark.anyio
async def test_scan_returns_a_curve():
    """Opening the slit lets more through. If that is ever not true, something
    is wrong with the geometry, not with the beamline."""
    async with await _session() as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            loaded = await session.call_tool(
                "load_beamline", {"name": "nsls2-hard-xray-fast"}
            )
            spec = json.loads(_content_text(loaded))
            result = await session.call_tool(
                "scan_element",
                {
                    "spec": spec, "element": "white_beam_slit",
                    "field": "opening_v_mm", "values": [0.2, 0.5, 1.0],
                    "metric": "transmission", "nrays": 4000,
                },
            )
    text = _content_text(result)
    assert len(_images(result)) == 1
    rows = [line.split("\t") for line in text.splitlines() if "\t" in line]
    values = [float(r[1]) for r in rows]
    assert values[0] < values[1] < values[2]


@pytest.mark.anyio
async def test_a_bad_element_name_explains_itself():
    async with await _session() as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            loaded = await session.call_tool(
                "load_beamline", {"name": "nsls2-hard-xray-fast"}
            )
            spec = json.loads(_content_text(loaded))
            result = await session.call_tool(
                "scan_element",
                {"spec": spec, "element": "nope", "field": "opening_v_mm",
                 "values": [1.0]},
            )
    text = _content_text(result)
    assert "No element called 'nope'" in text
    assert "white_beam_slit" in text
