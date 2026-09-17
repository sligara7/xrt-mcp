"""Running the trace and reducing it to the numbers a beamline scientist asks for.

This is the only module that touches xrt's beam arrays. Everything above it
sees `TraceResult`.

On absolute flux: xrt hands each ray of a synchrotron source a weight
`beam.sourceWeight` such that its own documented estimator

    flux  = sum(T_i * w_i)                 photons/second
    power = sum(T_i * E_i * e * w_i)       watts

holds for any downstream transmission factor T_i, which after propagation is
just the ray's surviving intensity Jss + Jpp. That is what this module
computes, so the numbers are xrt's, not a re-derivation of them.

Two things they are NOT. They are the flux *inside the sampled window* — the
energy range and the angular half-acceptance the source was given — so widening
`energy_min_eV`/`energy_max_eV` genuinely changes the answer. And a geometric
source carries no such weight at all, so flux comes back as None for one rather
than as a number that would look real.
"""

from __future__ import annotations

import io
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field

import numpy as np

from .spec import BeamlineSpec, DCM, Aperture, Mirror, Screen, Undulator, build

_ELEMENTARY_CHARGE = 1.602176634e-19  # J per eV, i.e. watts per eV/second


@contextmanager
def quiet():
    """Swallow xrt's progress chatter.

    The stdio transport in mcp 2.x already points fd 1 at stderr while it is
    serving, so a stray print cannot corrupt the protocol. This is about not
    filling the client's log with ten thousand ray counts.
    """
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        yield buf


@dataclass
class ScreenReading:
    """What the beam looked like where somebody put a screen."""

    name: str
    at_m: float
    rays_alive: int
    transmission: float  # fraction of the source's intensity that got here
    flux_ph_s: float | None
    power_W: float | None
    x_fwhm_um: float | None
    z_fwhm_um: float | None
    x_rms_um: float | None
    z_rms_um: float | None
    mean_energy_eV: float | None
    energy_fwhm_eV: float | None
    # kept for the renderer, not part of the tool's JSON reply
    _x_mm: np.ndarray | None = field(default=None, repr=False)
    _z_mm: np.ndarray | None = field(default=None, repr=False)
    _E_eV: np.ndarray | None = field(default=None, repr=False)
    _I: np.ndarray | None = field(default=None, repr=False)

    def summary(self) -> dict:
        """The JSON-safe part, which is what goes back over MCP."""
        return {
            k: v for k, v in self.__dict__.items() if not k.startswith("_")
        }


@dataclass
class TraceResult:
    beamline: str
    rays_shone: int
    source_flux_ph_s: float | None
    source_power_W: float | None
    readings: list[ScreenReading]
    warnings: list[str] = field(default_factory=list)

    def by_name(self, name: str) -> ScreenReading | None:
        for r in self.readings:
            if r.name == name:
                return r
        return None

    def summary(self) -> dict:
        return {
            "beamline": self.beamline,
            "rays_shone": self.rays_shone,
            "source_flux_ph_s": self.source_flux_ph_s,
            "source_power_W": self.source_power_W,
            "screens": [r.summary() for r in self.readings],
            "warnings": self.warnings,
        }


def _fwhm(values: np.ndarray, weights: np.ndarray, bins: int = 64) -> float | None:
    """FWHM of a weighted distribution, read off a histogram.

    Deliberately not 2.355*sigma: a slit-defined beam is a flat top, and the
    gaussian conversion would overstate it by a third.
    """
    if values.size < 20:
        return None
    hist, edges = np.histogram(values, bins=bins, weights=weights)
    if hist.max() <= 0:
        return None
    half = hist.max() / 2.0
    above = np.nonzero(hist >= half)[0]
    if above.size == 0:
        return None
    return float(edges[above[-1] + 1] - edges[above[0]])


def trace(spec: BeamlineSpec, nrays: int = 20000, seed: int | None = 1) -> TraceResult:
    """Shine the source, walk the beam through the beamline, read the screens."""
    if seed is not None:
        np.random.seed(seed)

    warnings: list[str] = []
    with quiet():
        built = build(spec, nrays=nrays)
        beam = built.source.shine()

        shone = int(beam.x.size)
        i_source = float(np.sum(beam.Jss + beam.Jpp))
        # xrt's own per-ray weight; absent on a geometric source
        weight = getattr(beam, "sourceWeight", None)
        if not isinstance(spec.source, Undulator):
            weight = None
        source_flux = i_source * weight if weight else None
        source_power = (
            float(np.sum((beam.Jss + beam.Jpp) * beam.E) * _ELEMENTARY_CHARGE * weight)
            if weight
            else None
        )

        readings: list[ScreenReading] = []
        for placed in built.placed:
            el, obj = placed.spec, placed.obj
            if isinstance(el, Aperture):
                obj.propagate(beam)
            elif isinstance(el, DCM):
                beam = obj.double_reflect(beam)[0]
            elif isinstance(el, Mirror):
                beam = obj.reflect(beam)[0]
            elif isinstance(el, Screen):
                readings.append(_read(obj, el, beam, i_source, weight))

    for r in readings:
        if r.rays_alive < 100:
            warnings.append(
                f"only {r.rays_alive} rays reached {r.name} — the numbers there are "
                f"noisy; raise nrays or open the apertures upstream"
            )
    if not readings:
        warnings.append("no screens in this beamline, so there is nothing to report")

    return TraceResult(
        beamline=spec.name,
        rays_shone=shone,
        source_flux_ph_s=source_flux,
        source_power_W=source_power,
        readings=readings,
        warnings=warnings,
    )


def _read(screen, el: Screen, beam, i_source: float, weight: float | None):
    exposed = screen.expose(beam)
    alive = exposed.state == 1
    n = int(alive.sum())
    if n == 0:
        return ScreenReading(
            name=el.name, at_m=el.at_m, rays_alive=0, transmission=0.0,
            flux_ph_s=0.0 if weight else None, power_W=0.0 if weight else None,
            x_fwhm_um=None, z_fwhm_um=None, x_rms_um=None, z_rms_um=None,
            mean_energy_eV=None, energy_fwhm_eV=None,
        )

    x, z, E = exposed.x[alive], exposed.z[alive], exposed.E[alive]
    I = exposed.Jss[alive] + exposed.Jpp[alive]
    i_here = float(np.sum(I))

    return ScreenReading(
        name=el.name,
        at_m=el.at_m,
        rays_alive=n,
        transmission=i_here / i_source if i_source > 0 else 0.0,
        flux_ph_s=(i_here * weight) if weight else None,
        power_W=(
            float(np.sum(I * E) * _ELEMENTARY_CHARGE * weight) if weight else None
        ),
        x_fwhm_um=_um(_fwhm(x, I)),
        z_fwhm_um=_um(_fwhm(z, I)),
        x_rms_um=_um(_weighted_rms(x, I)),
        z_rms_um=_um(_weighted_rms(z, I)),
        mean_energy_eV=float(np.average(E, weights=I)),
        energy_fwhm_eV=_fwhm(E, I),
        _x_mm=x, _z_mm=z, _E_eV=E, _I=I,
    )


def _weighted_rms(v: np.ndarray, w: np.ndarray) -> float:
    mean = np.average(v, weights=w)
    return float(np.sqrt(np.average((v - mean) ** 2, weights=w)))


def _um(mm: float | None) -> float | None:
    return None if mm is None else float(mm) * 1000.0
