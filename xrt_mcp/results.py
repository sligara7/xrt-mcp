"""What the tools return, declared.

Every tool used to be annotated `-> dict` or `-> list[dict]`, which had two
consequences nobody chose. A `dict` carries no field information, so the SDK
published no output schema at all and a model had to call a tool to learn what
came back. A `list[dict]` is not an object, so the SDK wrapped it in
`{"result": ...}` while the dicts went back bare — the inconsistent envelope the
first outside trial reported, which was never a style choice but a side effect
of the return annotation.

Declaring the results fixes both, and buys the thing that mattered more: a place
for the response contract to live. Every result that carries a number carries
`meta`, so a number can be traced to the build that produced it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .meta import Provenance, provenance


class Result(BaseModel):
    """Base for anything whose numbers somebody might write down."""

    meta: Provenance = Field(default_factory=provenance)


# --------------------------------------------------------------------------
# starting points
# --------------------------------------------------------------------------


class BeamlineSummary(BaseModel):
    name: str
    description: str


class BeamlineListing(Result):
    beamlines: list[BeamlineSummary]


# --------------------------------------------------------------------------
# optics, answered without tracing
# --------------------------------------------------------------------------


class CrystalAtEnergy(Result):
    crystal: str
    energy_eV: float
    d_spacing_A: float
    bragg_angle_deg: float
    darwin_width_urad: float
    energy_resolution_dE_over_E: float
    energy_bandwidth_eV: float


class CrystalEnergyRange(Result):
    crystal: str
    d_spacing_A: float
    min_energy_eV: float
    note: str


class AbsorptionEdge(BaseModel):
    """A step down in reflectivity inside the mirror's working range."""

    energy_eV: float
    reflectivity_below: float
    reflectivity_above: float


class ReflectivityCurve(BaseModel):
    energy_eV: list[float]
    reflectivity: list[float]
    points: int
    note: str


class MirrorCutoff(Result):
    coating: str
    pitch_mrad: float
    cutoff_energy_eV: float | None
    cutoff_definition: str
    absorption_edges: list[AbsorptionEdge]
    edge_definition: str
    reflectivity_curve: ReflectivityCurve | None = None


class Harmonic(BaseModel):
    n: int
    energy_eV: float


class UndulatorHarmonics(Result):
    period_mm: float
    K: float
    ring_GeV: float
    harmonics: list[Harmonic]


# --------------------------------------------------------------------------
# tracing
# --------------------------------------------------------------------------


class SampledWindow(BaseModel):
    """The window the source was sampled over.

    Flux and power are only ever the flux and power INSIDE this window, so it
    travels with them. The server instructions ask a caller to report the window
    it used; carrying it here makes it hard to quote the number without it.
    """

    energy_min_eV: float
    energy_max_eV: float
    x_prime_max_mrad: float | None = Field(
        None, description="horizontal half-acceptance; None for a geometric source"
    )
    z_prime_max_mrad: float | None = None


class ScreenReadingOut(BaseModel):
    name: str
    at_m: float
    rays_alive: int
    transmission: float = Field(
        description="fraction of the SOURCE's intensity surviving to this screen"
    )
    flux_ph_s: float | None
    power_W: float | None
    null_reason: str | None = Field(
        None,
        description="why flux and power are null, when they are — so a caller can "
        "tell 'this source carries no flux weight' from 'the trace failed'",
    )
    x_fwhm_um: float | None
    z_fwhm_um: float | None
    x_rms_um: float | None
    z_rms_um: float | None
    mean_energy_eV: float | None
    energy_fwhm_eV: float | None


class TraceOut(Result):
    beamline: str
    rays_shone: int
    seconds: float
    source_flux_ph_s: float | None
    source_power_W: float | None
    null_reason: str | None = None
    sampled_window: SampledWindow
    size_convention: str = (
        "beam sizes are FWHM and rms in microns, measured on the intensity-weighted "
        "distribution at the screen"
    )
    screens: list[ScreenReadingOut]
    warnings: list[str]
