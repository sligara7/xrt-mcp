"""The MCP surface: what an agent sees.

This module knows about MCP and about the shape of a beamline. It knows nothing
about ray tracing — that is `tracing`, which it calls — and nothing about
matplotlib — that is `render`. Keeping those apart is the whole reason this is
five small modules and not one big one: xrt 2.0 is beta and moving, and when it
moves, the change should land in one place.
"""

from __future__ import annotations

import copy
import time
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import Image, MCPServer
from pydantic import Field

from . import reference, render, templates
from .spec import BeamlineSpec
from .tracing import trace

INSTRUCTIONS = """\
xrt-mcp puts the xrt ray-tracing library behind MCP, for modelling synchrotron
beamlines: sources, optics and the beam that reaches the sample.

How to use it:

1. `list_beamlines` and `load_beamline` give you a working beamline to start
   from. Edit the spec it returns and pass it back — it is plain data.
2. `trace_beamline` propagates rays through it and returns the numbers: beam
   size at every screen, energy bandwidth, transmission, and for an undulator
   source, absolute flux in photons/second and power in watts.
3. `beam_image` returns the same trace as a picture.
4. `scan_element` sweeps one number — a slit gap, the monochromator energy, a
   mirror pitch — and gives you the curve.
5. The `crystal_*`, `mirror_cutoff` and `undulator_harmonics` tools answer
   optics questions instantly, without tracing. Use them to sanity-check a
   design before spending a minute on a trace.

Two things worth knowing before you read a number off this server:

- An undulator trace costs 20-60 seconds. A geometric source costs under a
  second and is physically a stand-in for the emission only; the optics are
  identical. Scan with the geometric one, confirm with the undulator one.
- Flux and power are within the SAMPLED window — the source's energy range and
  angular half-acceptance. Widening `energy_min_eV`/`energy_max_eV` genuinely
  changes the answer, so say what window you used when you report a number.
"""

mcp = MCPServer(
    name="xrt-mcp",
    instructions=INSTRUCTIONS,
    version="0.1.0",
)

Spec = Annotated[BeamlineSpec, Field(description="a beamline, as data")]
NRays = Annotated[
    int,
    Field(
        20000,
        ge=500,
        le=200000,
        description="rays to shine; more is less noisy and slower",
    ),
]


# --------------------------------------------------------------------------
# starting points
# --------------------------------------------------------------------------


@mcp.tool()
def list_beamlines() -> list[dict]:
    """List the ready-made beamlines you can start from."""
    return templates.listing()


@mcp.tool()
def load_beamline(
    name: Annotated[str, Field(description="one of the names from list_beamlines")],
    energy_eV: Annotated[
        float | None,
        Field(None, description="retune the beamline to this energy, where it applies"),
    ] = None,
) -> dict:
    """Load a ready-made beamline as an editable spec.

    Change anything in what comes back and pass it to `trace_beamline`. The
    undulator is tuned to the energy you ask for, so its gap follows the
    monochromator rather than being left where it was.
    """
    kwargs: dict[str, Any] = {}
    if energy_eV is not None:
        kwargs["energy_eV"] = energy_eV
    try:
        spec = templates.get(name, **kwargs)
    except TypeError:
        spec = templates.get(name)  # template takes no energy
    return spec.model_dump()


# --------------------------------------------------------------------------
# tracing
# --------------------------------------------------------------------------


@mcp.tool()
def trace_beamline(spec: Spec, nrays: NRays = 20000) -> dict:
    """Trace a beamline and return what reaches each screen.

    Beam sizes are FWHM and rms in microns, energies in eV. `transmission` is
    the fraction of the source's intensity that survives to that screen.
    `flux_ph_s` and `power_W` are absolute, and are None for a geometric
    source, which carries no flux normalisation.

    An undulator source takes 20-60 seconds. Warnings come back in `warnings`
    rather than as an error — a beamline that passes almost nothing is a real
    answer about that beamline.
    """
    started = time.monotonic()
    result = trace(spec, nrays=nrays)
    out = result.summary()
    out["seconds"] = round(time.monotonic() - started, 1)
    return out


@mcp.tool()
def beam_image(
    spec: Spec,
    screen: Annotated[
        str | None,
        Field(None, description="which screen; the last one if you leave it out"),
    ] = None,
    kind: Annotated[
        Literal["footprint", "spectrum", "layout"],
        Field("footprint", description="the beam at a screen, its spectrum, or "
                                       "beam size along the whole beamline"),
    ] = "footprint",
    nrays: NRays = 20000,
) -> list[str | Image]:
    """Trace a beamline and return a picture of the result, plus the numbers.

    `footprint` is the beam where it lands, with profiles through it.
    `spectrum` is what energies got through. `layout` plots beam size against
    distance along the beamline, which is where a focus shows up as a waist.
    """
    result = trace(spec, nrays=nrays)
    if not result.readings:
        return ["This beamline has no screens, so there is nothing to show. "
                "Add a Screen element where you want to look at the beam."]

    if kind == "layout":
        png = render.layout(result, spec)
        return [Image(data=png, format="png"), _describe(result)]

    reading = result.by_name(screen) if screen else result.readings[-1]
    if reading is None:
        have = ", ".join(r.name for r in result.readings)
        return [f"No screen called {screen!r}. This beamline has: {have}."]

    png = render.footprint(reading) if kind == "footprint" else render.spectrum(reading)
    return [Image(data=png, format="png"), _describe(result)]


@mcp.tool()
def scan_element(
    spec: Spec,
    element: Annotated[str, Field(description="the name of the element to vary")],
    field: Annotated[str, Field(description="the numeric field on it, e.g. "
                                            "'opening_v_mm', 'energy_eV', 'pitch_mrad'")],
    values: Annotated[list[float], Field(description="values to sweep", max_length=40)],
    metric: Annotated[
        Literal["flux_ph_s", "power_W", "transmission", "x_fwhm_um", "z_fwhm_um",
                "energy_fwhm_eV", "rays_alive"],
        Field("transmission", description="what to plot against the swept value"),
    ] = "transmission",
    screen: Annotated[
        str | None, Field(None, description="which screen to read; the last one by default")
    ] = None,
    nrays: NRays = 10000,
) -> list[str | Image]:
    """Sweep one number through a beamline and return the curve.

    This is the tool for questions of the form "what does the slit gap do to
    the flux" or "how does the spot grow as I detune the mirror". One trace per
    value, so use a geometric source unless you have the minutes: 10 values on
    an undulator is ten minutes, 10 values on a geometric source is seconds.
    """
    target = next((e for e in spec.elements if e.name == element), None)
    if target is None:
        have = ", ".join(e.name for e in spec.elements)
        return [f"No element called {element!r}. This beamline has: {have}."]
    if not hasattr(target, field):
        have = ", ".join(k for k in type(target).model_fields if k not in ("kind", "name"))
        return [f"{element!r} has no field {field!r}. It has: {have}."]

    xs: list[float] = []
    ys: list[float | None] = []
    for value in values:
        probe = copy.deepcopy(spec)
        element_copy = next(e for e in probe.elements if e.name == element)
        setattr(element_copy, field, value)
        result = trace(probe, nrays=nrays)
        reading = result.by_name(screen) if screen else (
            result.readings[-1] if result.readings else None
        )
        xs.append(value)
        ys.append(getattr(reading, metric) if reading else None)

    plotted = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if not plotted:
        return [f"Every point in this scan came back empty at {screen or 'the last screen'} — "
                f"nothing reached it. Check the apertures upstream."]

    png = render.scan_curve(
        [p[0] for p in plotted], [p[1] for p in plotted],
        x_label=f"{element}.{field}", y_label=metric,
        title=f"{spec.name} — {metric} at {screen or 'last screen'}",
    )
    table = "\n".join(f"  {x:g}\t{y!r}" for x, y in zip(xs, ys))
    return [
        Image(data=png, format="png"),
        f"{element}.{field} vs {metric}:\n{table}",
    ]


# --------------------------------------------------------------------------
# optics questions, answered without tracing
# --------------------------------------------------------------------------


@mcp.tool()
def crystal_at_energy(
    crystal: Annotated[Literal["Si111", "Si220", "Si311", "Si333"], Field("Si111")] = "Si111",
    energy_eV: Annotated[float, Field(9000.0, gt=0)] = 9000.0,
) -> dict:
    """Bragg angle, Darwin width and intrinsic energy resolution of a crystal.

    Instant. Worth calling before a trace: it tells you the bandwidth a
    monochromator can deliver, so you know what the trace ought to produce.
    """
    return reference.crystal_at_energy(crystal, energy_eV)


@mcp.tool()
def crystal_energy_range(
    crystal: Annotated[Literal["Si111", "Si220", "Si311", "Si333"], Field("Si111")] = "Si111",
) -> dict:
    """The energies a crystal can reach at all."""
    return reference.crystal_energy_range(crystal)


@mcp.tool()
def mirror_cutoff(
    coating: Annotated[Literal["Si", "Rh", "Pt", "Ir", "Au", "C"], Field("Rh")] = "Rh",
    pitch_mrad: Annotated[float, Field(3.0, gt=0, le=50)] = 3.0,
) -> dict:
    """The critical energy of a mirror at a grazing angle, and its reflectivity
    curve. This is how a mirror is used to kill undulator harmonics: set the
    cutoff between the harmonic you want and the one you don't."""
    return reference.mirror_cutoff(coating, pitch_mrad)


@mcp.tool()
def undulator_harmonics(
    period_mm: Annotated[float, Field(21.0, gt=0)] = 21.0,
    K: Annotated[float, Field(1.5, gt=0)] = 1.5,
    ring_GeV: Annotated[float, Field(3.0, gt=0)] = 3.0,
) -> dict:
    """Where an undulator's odd harmonics sit, for a given gap (K).

    NSLS-II is a 3 GeV ring, which is the default here.
    """
    return reference.undulator_harmonics(period_mm, K, ring_GeV)


def _describe(result) -> str:
    lines = [f"{result.beamline} — {result.rays_shone} rays"]
    if result.source_flux_ph_s is not None:
        lines.append(
            f"source: {result.source_flux_ph_s:.3g} ph/s, "
            f"{result.source_power_W:.3g} W in the sampled window"
        )
    for r in result.readings:
        bits = [f"{r.name} at {r.at_m:g} m:"]
        if r.rays_alive == 0:
            bits.append("nothing got here")
        else:
            bits.append(
                f"{_f(r.x_fwhm_um)} x {_f(r.z_fwhm_um)} um FWHM, "
                f"{r.mean_energy_eV:.1f} eV +/- {_f(r.energy_fwhm_eV, 2)} eV, "
                f"{r.transmission * 100:.3g}% transmitted"
            )
            if r.flux_ph_s is not None:
                bits.append(f"{r.flux_ph_s:.3g} ph/s")
            if r.power_W is not None:
                bits.append(f"{r.power_W:.3g} W")
        lines.append("  " + " ".join(bits))
    lines.extend(f"  ! {w}" for w in result.warnings)
    return "\n".join(lines)


def _f(value, places: int = 1) -> str:
    return "?" if value is None else f"{value:.{places}f}"
