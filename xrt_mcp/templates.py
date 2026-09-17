"""Beamlines an agent can start from.

These are deliberately shaped like real NSLS-II hard x-ray beamlines — a 3 GeV,
500 mA ring, an in-vacuum undulator, a white-beam slit, a Si(111) monochromator
and a toroidal focusing mirror — so that somebody who works on one recognises it
immediately. They are NOT models of any particular beamline: the distances and
apertures are plausible round numbers, not as-built values.

An agent reads one, changes what it likes, and traces the result. Nothing here
is privileged; a template is just a spec that was written down in advance.
"""

from __future__ import annotations

from .spec import Aperture, BeamlineSpec, DCM, GeometricSource, Mirror, Ring, Screen, Undulator


def nsls2_hard_xray(energy_eV: float = 9000.0, fast: bool = False) -> BeamlineSpec:
    """Undulator -> white-beam slit -> Si(111) DCM -> toroidal mirror -> sample.

    The undulator is tuned so its 5th harmonic lands on `energy_eV`, and the
    sampled energy window is a narrow band around it: outside that band the
    monochromator would throw the rays away anyway, and sampling there only buys
    noise. `fast=True` swaps in a geometric source — same optics, no emission
    physics, about a hundred times quicker.
    """
    half_window = 25.0
    if fast:
        source = GeometricSource(
            energy_min_eV=energy_eV - half_window,
            energy_max_eV=energy_eV + half_window,
        )
    else:
        source = Undulator(
            tune_to_eV=energy_eV,
            harmonic=5,
            energy_min_eV=energy_eV - half_window,
            energy_max_eV=energy_eV + half_window,
        )
    return BeamlineSpec(
        name=f"NSLS-II-like hard x-ray, {energy_eV:.0f} eV",
        source=source,
        elements=[
            Screen(name="front_end", at_m=25.0),
            Aperture(name="white_beam_slit", at_m=25.5, opening_h_mm=1.0, opening_v_mm=1.0),
            DCM(name="dcm", at_m=30.0, crystal="Si111", energy_eV=energy_eV,
                fixed_offset_mm=20.0),
            Screen(name="after_dcm", at_m=31.0),
            Mirror(name="focusing_mirror", at_m=34.0, coating="Rh", pitch_mrad=3.0,
                   deflect="down", focus_at_m=11.0, length_mm=400.0, width_mm=30.0),
            Screen(name="sample", at_m=45.0),
        ],
    )


def nsls2_white_beam() -> BeamlineSpec:
    """Undulator straight into the front end, nothing in the way.

    What this is for is the power number: the watts landing on the first
    aperture is the question that decides whether a front end survives, and it
    is answerable before any optics exist.
    """
    return BeamlineSpec(
        name="NSLS-II-like white beam onto the front end",
        source=Undulator(
            tune_to_eV=9000.0, harmonic=5,
            energy_min_eV=2000.0, energy_max_eV=30000.0,
            x_prime_max_mrad=0.15, z_prime_max_mrad=0.15,
        ),
        elements=[
            Screen(name="front_end", at_m=25.0),
            Aperture(name="fe_mask", at_m=25.5, opening_h_mm=2.0, opening_v_mm=1.0),
            Screen(name="after_mask", at_m=26.0),
        ],
    )


def bending_magnet_style_geometric() -> BeamlineSpec:
    """A wide, divergent geometric source through a slit and a mono — the shape
    of a bending-magnet beamline, without the emission physics. Fast enough to
    scan."""
    return BeamlineSpec(
        name="wide-source beamline (geometric)",
        source=GeometricSource(
            size_x_um=120.0, size_z_um=25.0,
            divergence_x_mrad=0.5, divergence_z_mrad=0.1,
            energy_min_eV=7000.0, energy_max_eV=11000.0,
        ),
        elements=[
            Aperture(name="slit", at_m=15.0, opening_h_mm=2.0, opening_v_mm=1.0),
            DCM(name="dcm", at_m=20.0, crystal="Si111", energy_eV=9000.0),
            Screen(name="sample", at_m=25.0),
        ],
    )


TEMPLATES = {
    "nsls2-hard-xray": (
        "NSLS-II-like undulator beamline: white-beam slit, Si(111) DCM, "
        "toroidal focusing mirror, sample at 45 m. The main demo.",
        nsls2_hard_xray,
    ),
    "nsls2-hard-xray-fast": (
        "The same optics with a geometric source instead of undulator emission "
        "- about a hundred times quicker, so it is the one to scan with.",
        lambda energy_eV=9000.0: nsls2_hard_xray(energy_eV, fast=True),
    ),
    "nsls2-white-beam": (
        "Undulator straight onto the front-end mask, wide open in energy and "
        "angle. Answers the heat-load question: how many watts land there.",
        nsls2_white_beam,
    ),
    "wide-source-geometric": (
        "A wide, divergent source through a slit and a Si(111) mono. "
        "Bending-magnet-shaped, fast.",
        bending_magnet_style_geometric,
    ),
}


def get(name: str, **kwargs) -> BeamlineSpec:
    if name not in TEMPLATES:
        raise KeyError(
            f"no template called {name!r}. Available: {', '.join(sorted(TEMPLATES))}"
        )
    return TEMPLATES[name][1](**kwargs)


def listing() -> list[dict]:
    return [
        {"name": n, "description": d} for n, (d, _) in sorted(TEMPLATES.items())
    ]
