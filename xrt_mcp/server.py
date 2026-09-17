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

from . import __version__, reference, render, templates
from .meta import ServerInfo
from .meta import server_info as server_info_impl
from .results import (
    BeamlineListing,
    BeamlineSummary,
    CrystalAtEnergy,
    CrystalEnergyRange,
    MirrorCutoff,
    SampledWindow,
    ScreenReadingOut,
    TraceOut,
    UndulatorHarmonics,
)
from .spec import BeamlineSpec, GeometricSource, Undulator
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
    version=__version__,
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
def list_beamlines() -> BeamlineListing:
    """List the ready-made beamlines you can start from."""
    return BeamlineListing(
        beamlines=[BeamlineSummary(**t) for t in templates.listing()]
    )


@mcp.tool()
def load_beamline(
    name: Annotated[str, Field(description="one of the names from list_beamlines")],
    energy_eV: Annotated[
        float | None,
        Field(None, description="retune the beamline to this energy, where it applies"),
    ] = None,
) -> BeamlineSpec:
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
    return spec


# --------------------------------------------------------------------------
# tracing
# --------------------------------------------------------------------------


@mcp.tool()
def server_info() -> ServerInfo:
    """Which build is answering: this server, the xrt it drives, and the stack
    underneath. Every answer that carries a number also carries the first two in
    its `meta` block; this is here for the rest, which do not change within a
    session and would be wasted bytes on every reply."""
    return server_info_impl()


@mcp.tool()
def trace_beamline(spec: Spec, nrays: NRays = 20000) -> TraceOut:
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
    reason = None if result.source_flux_ph_s is not None else _GEOMETRIC_NULL
    return TraceOut(
        beamline=result.beamline,
        rays_shone=result.rays_shone,
        seconds=round(time.monotonic() - started, 1),
        source_flux_ph_s=result.source_flux_ph_s,
        source_power_W=result.source_power_W,
        null_reason=reason,
        sampled_window=_sampled_window(spec),
        screens=[
            ScreenReadingOut(**r.summary(), null_reason=reason)
            for r in result.readings
        ],
        warnings=result.warnings,
    )


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
    energy_eV: Annotated[
        float,
        Field(gt=0, description="the energy you actually care about — required, "
                                "because a defaulted energy would answer a question "
                                "you did not ask and look exactly like an answer"),
    ],
    crystal: Annotated[Literal["Si111", "Si220", "Si311", "Si333"], Field("Si111")] = "Si111",
) -> CrystalAtEnergy:
    """Bragg angle, Darwin width and intrinsic energy resolution of a crystal.

    Instant. Worth calling before a trace: it tells you the bandwidth a
    monochromator can deliver, so you know what the trace ought to produce.
    """
    return CrystalAtEnergy(**reference.crystal_at_energy(crystal, energy_eV))


@mcp.tool()
def crystal_energy_range(
    crystal: Annotated[Literal["Si111", "Si220", "Si311", "Si333"], Field("Si111")] = "Si111",
) -> CrystalEnergyRange:
    """The energies a crystal can reach at all."""
    return CrystalEnergyRange(**reference.crystal_energy_range(crystal))


@mcp.tool()
def mirror_cutoff(
    pitch_mrad: Annotated[float, Field(gt=0, le=50, description="grazing angle — required")],
    coating: Annotated[Literal["Si", "Rh", "Pt", "Ir", "Au", "C"], Field("Rh")] = "Rh",
    include_curve: Annotated[
        bool,
        Field(False, description="return the full reflectivity curve as well — about "
                                 "400 points and 30 KB, so ask for it when you want to "
                                 "plot it, not to read a number off it"),
    ] = False,
) -> MirrorCutoff:
    """The critical energy of a mirror at a grazing angle, and the coating's
    absorption edges below it.

    This is how a mirror kills undulator harmonics: set the cutoff between the
    harmonic you want and the one you don't. The edges matter for the same job
    and are easier to miss — an edge drops reflectivity in a band well below the
    cutoff and it does NOT recover afterwards. Rh at 3 mrad is the standard
    case: its three L edges take it from 0.97 to 0.89 between 3.0 and 3.4 keV,
    and it is still only 0.91 at 4.2 keV."""
    return MirrorCutoff(**reference.mirror_cutoff(coating, pitch_mrad, include_curve))


@mcp.tool()
def undulator_harmonics(
    period_mm: Annotated[float, Field(gt=0, description="undulator period — required")],
    K: Annotated[float, Field(gt=0, description="deflection parameter — required")],
    ring_GeV: Annotated[
        float,
        Field(gt=0, description="storage ring energy — required. NSLS-II is 3.0; it is "
                                "not defaulted because the harmonic energies go as the "
                                "SQUARE of it, so a wrong guess here is not a small error"),
    ],
) -> UndulatorHarmonics:
    """Where an undulator's odd harmonics sit, for a given gap (K)."""
    return UndulatorHarmonics(**reference.undulator_harmonics(period_mm, K, ring_GeV))


_GEOMETRIC_NULL = (
    "a geometric source carries no flux normalisation, so there is no absolute flux "
    "or power to report. This is a property of the source you chose, not a failed "
    "calculation — swap in an undulator source if you need absolute numbers."
)


def _sampled_window(spec: BeamlineSpec) -> SampledWindow:
    """The window the source was sampled over, travelling with the numbers it
    bounds. Flux and power are only ever the flux and power inside it."""
    src = spec.source
    if isinstance(src, Undulator):
        return SampledWindow(
            energy_min_eV=src.energy_min_eV,
            energy_max_eV=src.energy_max_eV,
            x_prime_max_mrad=src.x_prime_max_mrad,
            z_prime_max_mrad=src.z_prime_max_mrad,
        )
    return SampledWindow(
        energy_min_eV=src.energy_min_eV, energy_max_eV=src.energy_max_eV
    )


def _describe(result) -> str:
    info = server_info_impl()
    lines = [
        f"{result.beamline} — {result.rays_shone} rays "
        f"(xrt-mcp {info.xrt_mcp}, xrt {info.xrt})"
    ]
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
