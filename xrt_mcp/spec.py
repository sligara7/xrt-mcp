"""The declarative beamline description an agent sends, and the builder that
turns one into live xrt objects.

Nothing in this module executes agent-supplied Python. An agent sends data; we
construct xrt objects from it ourselves. That is the whole point of the spec —
see the design note in README.md under "Why a spec and not a script".

Units, stated once so nobody has to guess:

===========================  ======================================
distance along the beamline  metres, from the source, along the beam
apertures, offsets, optics   millimetres
angles                       milliradians
energies                     electron-volts
===========================  ======================================

xrt itself works in millimetres and radians; the conversion happens here and
nowhere else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------------------
# the storage ring
# --------------------------------------------------------------------------


class Ring(BaseModel):
    """The electron beam the source sits in. Defaults are NSLS-II's low-beta
    straight: 3 GeV, 500 mA, 0.9 nm.rad horizontal emittance at 1% coupling."""

    name: str = "NSLS-II"
    energy_GeV: float = 3.0
    current_A: float = 0.5
    sigma_x_um: float = Field(40.0, description="horizontal electron beam size, rms")
    sigma_z_um: float = Field(3.0, description="vertical electron beam size, rms")
    emittance_x_nmrad: float = 0.9
    emittance_z_nmrad: float = 0.008


# --------------------------------------------------------------------------
# undulator tuning
#
# The resonance condition, in the form beamline people quote it:
#
#     E_n [keV] = 0.9496 * n * E_ring[GeV]^2 / (period[cm] * (1 + K^2 / 2))
# --------------------------------------------------------------------------

_RESONANCE = 0.9496  # keV.cm/GeV^2


def resonance_energy_eV(period_mm: float, K: float, ring_GeV: float, harmonic: int) -> float:
    """Where harmonic `n` of this undulator sits, in eV."""
    period_cm = period_mm / 10.0
    return 1000.0 * _RESONANCE * harmonic * ring_GeV**2 / (
        period_cm * (1.0 + K**2 / 2.0)
    )


def K_for_energy(energy_eV: float, period_mm: float, ring_GeV: float, harmonic: int) -> float:
    """The K that puts harmonic `n` at `energy_eV`. Raises if the undulator
    cannot reach it — which is real information, not a failure to work around:
    it means that energy needs a different harmonic or a different device."""
    period_cm = period_mm / 10.0
    factor = 1000.0 * _RESONANCE * harmonic * ring_GeV**2 / (period_cm * energy_eV)
    if factor <= 1.0:
        raise ValueError(
            f"harmonic {harmonic} of a {period_mm} mm undulator on a {ring_GeV} GeV "
            f"ring cannot reach {energy_eV} eV — its highest energy on that harmonic "
            f"is {resonance_energy_eV(period_mm, 0.0, ring_GeV, harmonic):.0f} eV, at "
            f"K -> 0. Try a higher harmonic."
        )
    return math.sqrt(2.0 * (factor - 1.0))


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------


class Undulator(BaseModel):
    """An undulator. The real thing: xrt computes the emission properly, which
    is why a trace takes tens of seconds rather than tenths.

    Give either `K` directly, or `tune_to_eV` and a `harmonic` and the gap is
    solved for you. Tuning matters more than it looks: an undulator set to the
    wrong K puts almost no flux at the energy you asked for, and the trace will
    come back honest and nearly empty rather than wrong."""

    kind: Literal["undulator"] = "undulator"
    name: str = "IVU21"
    period_mm: float = 21.0
    n_periods: int = 140
    K: float | None = Field(None, description="deflection parameter; the gap, in effect")
    tune_to_eV: float | None = Field(
        None, description="solve K so this energy lands on `harmonic`"
    )
    harmonic: int = 5
    ring: Ring = Field(default_factory=Ring)
    energy_min_eV: float = 8900.0
    energy_max_eV: float = 9100.0
    x_prime_max_mrad: float = Field(0.05, description="horizontal half-acceptance sampled")
    z_prime_max_mrad: float = Field(0.05, description="vertical half-acceptance sampled")

    @model_validator(mode="after")
    def _resolve_K(self):
        if self.K is None and self.tune_to_eV is None:
            raise ValueError("give the undulator either K or tune_to_eV")
        if self.tune_to_eV is not None:
            self.K = K_for_energy(
                self.tune_to_eV, self.period_mm, self.ring.energy_GeV, self.harmonic
            )
        return self

    def resonance_eV(self, harmonic: int | None = None) -> float:
        return resonance_energy_eV(
            self.period_mm, self.K, self.ring.energy_GeV, harmonic or self.harmonic
        )


class GeometricSource(BaseModel):
    """A gaussian pencil with no emission physics. Fast — milliseconds, not
    tens of seconds — so it is what makes a multi-point scan demonstrable.
    Optics, apertures and the monochromator behave exactly as they do with a
    real source; only the source itself is a stand-in."""

    kind: Literal["geometric"] = "geometric"
    name: str = "geometric"
    size_x_um: float = 40.0
    size_z_um: float = 3.0
    divergence_x_mrad: float = 0.03
    divergence_z_mrad: float = 0.01
    energy_min_eV: float = 8900.0
    energy_max_eV: float = 9100.0


Source = Annotated[Union[Undulator, GeometricSource], Field(discriminator="kind")]


# --------------------------------------------------------------------------
# elements
# --------------------------------------------------------------------------


class Aperture(BaseModel):
    """A slit. Centred on the beam."""

    kind: Literal["aperture"] = "aperture"
    name: str
    at_m: float
    opening_h_mm: float = 1.0
    opening_v_mm: float = 1.0


class DCM(BaseModel):
    """A double crystal monochromator. Aligned to `energy_eV`; the second
    crystal carries the beam back parallel, `fixed_offset_mm` higher."""

    kind: Literal["dcm"] = "dcm"
    name: str = "DCM"
    at_m: float
    crystal: Literal["Si111", "Si220", "Si311", "Si333"] = "Si111"
    energy_eV: float
    fixed_offset_mm: float = 20.0


class Mirror(BaseModel):
    """A grazing-incidence mirror. Give `focus_at_m` and it becomes a toroid
    whose radii are solved from the Coddington equations for that focus;
    leave it out and it stays flat."""

    kind: Literal["mirror"] = "mirror"
    name: str
    at_m: float
    coating: Literal["Si", "Rh", "Pt", "Ir", "Au"] = "Rh"
    pitch_mrad: float = 3.0
    deflect: Literal["up", "down"] = "up"
    focus_at_m: float | None = Field(
        None, description="distance from THIS mirror to the focus, in metres"
    )
    length_mm: float = 400.0
    width_mm: float = 30.0


class Screen(BaseModel):
    """A place to look at the beam. Every screen you list produces numbers,
    and can produce an image."""

    kind: Literal["screen"] = "screen"
    name: str
    at_m: float


Element = Annotated[Union[Aperture, DCM, Mirror, Screen], Field(discriminator="kind")]


class BeamlineSpec(BaseModel):
    """A whole beamline. `elements` is sorted by `at_m` before tracing, so the
    order you list them in does not matter."""

    name: str = "beamline"
    source: Source
    elements: list[Element] = Field(default_factory=list)

    def ordered(self) -> list[Element]:
        return sorted(self.elements, key=lambda e: e.at_m)

    def screens(self) -> list[Screen]:
        return [e for e in self.ordered() if isinstance(e, Screen)]


# --------------------------------------------------------------------------
# building it
# --------------------------------------------------------------------------


@dataclass
class Placed:
    """One element, its xrt object, and where the beam actually was when it
    got there — height above the source plane and the angle it was travelling
    at. Tracking these is what lets a mirror and a monochromator coexist
    without the beam walking off everything downstream."""

    spec: Element
    obj: object
    y_mm: float
    z_mm: float
    theta_rad: float


@dataclass
class Built:
    beamline: object
    source: object
    placed: list[Placed]


_CRYSTALS = {
    "Si111": (1, 1, 1),
    "Si220": (2, 2, 0),
    "Si311": (3, 1, 1),
    "Si333": (3, 3, 3),
}

# g/cm3, for the coatings we offer
_COATING_RHO = {"Si": 2.33, "Rh": 12.41, "Pt": 21.45, "Ir": 22.42, "Au": 19.32}


def build(spec: BeamlineSpec, nrays: int = 20000) -> Built:
    """Turn a spec into live xrt objects, positioned along the real beam path."""
    import xrt.backends.raycing as raycing
    import xrt.backends.raycing.apertures as ra
    import xrt.backends.raycing.materials as rm
    import xrt.backends.raycing.oes as roe
    import xrt.backends.raycing.screens as rsc
    import xrt.backends.raycing.sources as rs

    bl = raycing.BeamLine(azimuth=0, height=0)
    src = _build_source(spec.source, bl, rs, nrays)

    s_mm = 0.0  # path length travelled so far
    y_mm = 0.0  # horizontal-plane coordinate, along the nominal axis
    z_mm = 0.0  # height
    theta = 0.0  # angle of travel in the vertical plane

    placed: list[Placed] = []
    for el in spec.ordered():
        d = el.at_m * 1000.0 - s_mm
        if d < 0:
            raise ValueError(
                f"{el.name} at {el.at_m} m is behind the previous element"
            )
        s_mm += d
        y_mm += d * math.cos(theta)
        z_mm += d * math.sin(theta)
        centre = (0.0, y_mm, z_mm)

        if isinstance(el, Aperture):
            h, v = el.opening_h_mm / 2.0, el.opening_v_mm / 2.0
            obj = ra.RectangularAperture(
                bl, el.name, centre,
                blades={"left": -h, "right": h, "bottom": -v, "top": v},
            )
        elif isinstance(el, Screen):
            obj = rsc.Screen(bl, el.name, centre)
        elif isinstance(el, DCM):
            hkl = _CRYSTALS[el.crystal]
            crystal = rm.CrystalSi(hkl=hkl, name=el.crystal)
            obj = roe.DCM(
                bl, el.name, centre,
                surface=(el.crystal,),
                material=(crystal,), material2=(crystal,),
                bragg=f"{el.energy_eV} eV",
                fixedOffset=el.fixed_offset_mm,
                limPhysX=(-15, 15), limPhysY=(-40, 40),
                limPhysX2=(-15, 15), limPhysY2=(-100, 100),
                alarmLevel=None,
            )
            z_mm += el.fixed_offset_mm  # the beam leaves higher, same direction
        elif isinstance(el, Mirror):
            alpha = el.pitch_mrad * 1e-3
            sign = 1.0 if el.deflect == "up" else -1.0
            # the surface has to tilt to meet a beam that may already be angled
            pitch = alpha + sign * theta if el.deflect == "up" else alpha - theta
            material = rm.Material(
                el.coating, rho=_COATING_RHO[el.coating], name=el.coating
            )
            kw = dict(
                surface=(el.coating,), material=(material,),
                pitch=pitch, positionRoll=0.0 if el.deflect == "up" else math.pi,
                limPhysX=(-el.width_mm / 2, el.width_mm / 2),
                limPhysY=(-el.length_mm / 2, el.length_mm / 2),
                alarmLevel=None,
            )
            if el.focus_at_m is None:
                obj = roe.OE(bl, el.name, centre, **kw)
            else:
                p_mm = s_mm  # source to mirror, along the beam
                q_mm = el.focus_at_m * 1000.0
                obj = roe.ToroidMirror(
                    bl, el.name, centre,
                    R=(p_mm, q_mm, alpha), r=(p_mm, q_mm, alpha), **kw
                )
            theta += sign * 2.0 * alpha
        else:  # pragma: no cover - the union is closed
            raise ValueError(f"unknown element kind: {el!r}")

        placed.append(Placed(spec=el, obj=obj, y_mm=y_mm, z_mm=z_mm, theta_rad=theta))

    return Built(beamline=bl, source=src, placed=placed)


def _build_source(s: Source, bl, rs, nrays: int):
    if isinstance(s, Undulator):
        r = s.ring
        return rs.Undulator(
            bl, name=s.name, center=(0, 0, 0), nrays=nrays,
            period=s.period_mm, n=s.n_periods, K=s.K,
            eE=r.energy_GeV, eI=r.current_A,
            eSigmaX=r.sigma_x_um, eSigmaZ=r.sigma_z_um,
            eEpsilonX=r.emittance_x_nmrad, eEpsilonZ=r.emittance_z_nmrad,
            eMin=s.energy_min_eV, eMax=s.energy_max_eV,
            xPrimeMax=s.x_prime_max_mrad, zPrimeMax=s.z_prime_max_mrad,
            distE="eV",
        )
    return rs.GeometricSource(
        bl, name=s.name, center=(0, 0, 0), nrays=nrays,
        dx=s.size_x_um / 1000.0, dz=s.size_z_um / 1000.0,
        dxprime=s.divergence_x_mrad * 1e-3, dzprime=s.divergence_z_mrad * 1e-3,
        distE="flat", energies=(s.energy_min_eV, s.energy_max_eV),
    )
