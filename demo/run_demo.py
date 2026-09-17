"""The demo, driven through a real MCP client.

Run it:

    .venv/bin/python demo/run_demo.py

It starts the server the way a client would, over stdio, and walks through the
questions a beamline scientist would actually ask — what can this monochromator
deliver, where do the harmonics sit, what reaches the sample, what does the
slit do to the flux. Images land in demo/output/.

Nothing here is privileged. Every call below is one an LLM can make on its own;
this script just fixes the order so the same thing happens every time.
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo" / "output"

FAST = "--fast" in sys.argv  # geometric source: seconds instead of minutes


def say(title: str, body: str = "") -> None:
    print(f"\n\033[1m{title}\033[0m")
    if body:
        print(textwrap.indent(body.rstrip(), "    "))


def text_of(result) -> str:
    return "\n".join(c.text for c in result.content if getattr(c, "type", None) == "text")


def save_images(result, stem: str) -> list[Path]:
    import base64

    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, block in enumerate(c for c in result.content if getattr(c, "type", None) == "image"):
        path = OUT / (f"{stem}.png" if i == 0 else f"{stem}-{i}.png")
        path.write_bytes(base64.b64decode(block.data))
        paths.append(path)
    return paths


async def main() -> None:
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "xrt_mcp"], cwd=str(ROOT)
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            info = await session.initialize()
            say("connected", f"{info.server_info.name} {info.server_info.version}")

            tools = (await session.list_tools()).tools
            say("tools on offer", "\n".join(f"{t.name}" for t in tools))

            # ---------------------------------------------------------------
            say("1. Before designing anything — what can Si(111) deliver at 9 keV?",
                "An instant question. No rays are traced to answer it.")
            r = await session.call_tool(
                "crystal_at_energy", {"crystal": "Si111", "energy_eV": 9000}
            )
            crystal = json.loads(text_of(r))
            say("", json.dumps(crystal, indent=2))

            # ---------------------------------------------------------------
            say("2. Where do the harmonics of a 21 mm in-vacuum undulator sit?",
                "The ring energy has to be given: harmonic energies go as its square.")
            r = await session.call_tool(
                "undulator_harmonics", {"period_mm": 21.0, "K": 1.588, "ring_GeV": 3.0}
            )
            harmonics = json.loads(text_of(r))
            say("", "\n".join(
                f"n={h['n']}: {h['energy_eV']:.0f} eV" for h in harmonics["harmonics"]
            ))

            # ---------------------------------------------------------------
            say("3. A Rh mirror at 3 mrad — where does it stop reflecting?",
                "This is how you kill the 7th harmonic while keeping the 5th.")
            r = await session.call_tool(
                "mirror_cutoff", {"coating": "Rh", "pitch_mrad": 3.0}
            )
            cutoff = json.loads(text_of(r))
            say("", f"cutoff {cutoff['cutoff_energy_eV']:.0f} eV "
                    f"({cutoff['cutoff_definition']})")
            say("", "and the edges in the coating below that cutoff, which are the "
                    "easy thing to miss:\n" + "\n".join(
                        f"  {ed['energy_eV']:7.0f} eV   R {ed['reflectivity_below']:.3f}"
                        f" -> {ed['reflectivity_above']:.3f}"
                        for ed in cutoff["absorption_edges"]
                    ))

            # ---------------------------------------------------------------
            name = "nsls2-hard-xray-fast" if FAST else "nsls2-hard-xray"
            say(f"4. Load a beamline: {name}",
                "Undulator, white-beam slit, Si(111) DCM, toroidal focusing mirror.")
            r = await session.call_tool("load_beamline", {"name": name, "energy_eV": 9000})
            spec = json.loads(text_of(r))
            say("", json.dumps(spec, indent=2)[:1400] + "\n...")

            # ---------------------------------------------------------------
            say("5. Trace it.",
                "Undulator emission takes about half a minute; --fast swaps in a "
                "geometric source.")
            r = await session.call_tool("trace_beamline", {"spec": spec, "nrays": 20000})
            traced = json.loads(text_of(r))
            say("", _table(traced))

            # ---------------------------------------------------------------
            say("6. Show the beam at the sample.")
            r = await session.call_tool(
                "beam_image",
                {"spec": spec, "screen": "sample", "kind": "footprint", "nrays": 20000},
            )
            for p in save_images(r, "sample-footprint"):
                say("", f"wrote {p.relative_to(ROOT)}")

            say("7. And the beam size along the whole beamline.")
            r = await session.call_tool(
                "beam_image", {"spec": spec, "kind": "layout", "nrays": 20000}
            )
            for p in save_images(r, "layout"):
                say("", f"wrote {p.relative_to(ROOT)}")

            # ---------------------------------------------------------------
            say("8. What does the white-beam slit do to the flux?",
                "One trace per point, so this one uses the fast source.")
            r = await session.call_tool("load_beamline", {"name": "nsls2-hard-xray-fast"})
            fast_spec = json.loads(text_of(r))
            r = await session.call_tool(
                "scan_element",
                {
                    "spec": fast_spec, "element": "white_beam_slit",
                    "field": "opening_v_mm",
                    "values": [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5],
                    "metric": "transmission", "screen": "sample", "nrays": 8000,
                },
            )
            say("", text_of(r))
            for p in save_images(r, "slit-scan"):
                say("", f"wrote {p.relative_to(ROOT)}")

            # ---------------------------------------------------------------
            say("9. Change the design and trace again.",
                "Detune the focusing mirror by half a milliradian — the spot should "
                "grow, because the mirror is no longer at the angle its radii were "
                "computed for.")
            mirror = next(e for e in fast_spec["elements"] if e["name"] == "focusing_mirror")
            mirror["pitch_mrad"] = 3.5
            r = await session.call_tool(
                "trace_beamline", {"spec": fast_spec, "nrays": 20000}
            )
            detuned = json.loads(text_of(r))
            say("", _table(detuned))

    say("done", f"images in {OUT.relative_to(ROOT)}")


def _table(traced: dict) -> str:
    head = f"{'screen':<16}{'at':>6}{'H FWHM':>10}{'V FWHM':>10}{'dE':>9}{'transm':>10}"
    if traced.get("source_flux_ph_s"):
        head += f"{'flux':>12}{'power':>10}"
    rows = [head, "-" * len(head)]
    for s in traced["screens"]:
        row = (f"{s['name']:<16}{s['at_m']:>5.0f}m"
               f"{_n(s['x_fwhm_um']):>10}{_n(s['z_fwhm_um']):>10}"
               f"{_n(s['energy_fwhm_eV'], 2):>9}{s['transmission'] * 100:>9.3g}%")
        if s.get("flux_ph_s") is not None:
            row += f"{s['flux_ph_s']:>12.2e}{s['power_W']:>10.2e}"
        rows.append(row)
    rows += [f"! {w}" for w in traced.get("warnings", [])]
    rows.append("")
    rows.append("H/V FWHM in microns, dE in eV, transm = fraction of source intensity")
    return "\n".join(rows)


def _n(value, places: int = 1) -> str:
    return "-" if value is None else f"{value:.{places}f}"


if __name__ == "__main__":
    asyncio.run(main())
