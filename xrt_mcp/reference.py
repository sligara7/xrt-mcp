"""Read-only questions about x-ray optics, answered straight out of xrt.

Nothing here traces anything, so everything here returns in milliseconds. That
matters more than it sounds: it means an agent can check that the optics it is
about to specify make sense before spending a minute on a trace.
"""

from __future__ import annotations

import math

import numpy as np

CRYSTALS = {
    "Si111": (1, 1, 1),
    "Si220": (2, 2, 0),
    "Si311": (3, 1, 1),
    "Si333": (3, 3, 3),
}

COATINGS = {"Si": 2.33, "Rh": 12.41, "Pt": 21.45, "Ir": 22.42, "Au": 19.32, "C": 2.2}


def crystal_at_energy(crystal: str, energy_eV: float) -> dict:
    """Bragg angle, Darwin width and the resolving power of a crystal at an
    energy — the three numbers that decide whether a monochromator choice is
    right before anything is built."""
    import xrt.backends.raycing.materials as rm

    if crystal not in CRYSTALS:
        raise KeyError(f"unknown crystal {crystal!r}; have {', '.join(CRYSTALS)}")
    xtal = rm.CrystalSi(hkl=CRYSTALS[crystal])
    theta = float(xtal.get_Bragg_angle(energy_eV))
    darwin = float(np.real(xtal.get_Darwin_width(energy_eV)))
    # dE/E = cot(theta) * dtheta, for a perfect crystal at its Darwin width
    resolution = darwin / math.tan(theta)
    return {
        "crystal": crystal,
        "energy_eV": energy_eV,
        "d_spacing_A": float(xtal.d),
        "bragg_angle_deg": math.degrees(theta),
        "darwin_width_urad": darwin * 1e6,
        "energy_resolution_dE_over_E": resolution,
        "energy_bandwidth_eV": resolution * energy_eV,
    }


def crystal_energy_range(crystal: str) -> dict:
    """The energies a crystal can reach at all — backscattering at the bottom,
    grazing at the top."""
    import xrt.backends.raycing.materials as rm

    xtal = rm.CrystalSi(hkl=CRYSTALS[crystal])
    e_min = float(xtal.get_backscattering_energy())
    return {
        "crystal": crystal,
        "d_spacing_A": float(xtal.d),
        "min_energy_eV": e_min,
        "note": (
            "min_energy_eV is backscattering, where the Bragg angle reaches 90 deg. "
            "There is no hard upper limit; the practical one is where the Bragg "
            "angle gets too shallow for the mechanics, usually around 3-5 deg."
        ),
    }


def mirror_cutoff(coating: str, pitch_mrad: float) -> dict:
    """The critical energy of a mirror at a grazing angle — above it the mirror
    stops reflecting, which is exactly how a mirror is used to kill harmonics."""
    import xrt.backends.raycing.materials as rm

    if coating not in COATINGS:
        raise KeyError(f"unknown coating {coating!r}; have {', '.join(COATINGS)}")
    material = rm.Material(coating, rho=COATINGS[coating], name=coating)
    theta = pitch_mrad * 1e-3
    energies = np.logspace(math.log10(1000.0), math.log10(60000.0), 400)
    # reflectivity at grazing incidence: beamInDotNormal = -sin(theta)
    amp_s, amp_p = material.get_amplitude(energies, -math.sin(theta))[:2]
    refl = np.abs(amp_s) ** 2
    below = np.nonzero(refl >= 0.5)[0]
    cutoff = float(energies[below[-1]]) if below.size else None
    return {
        "coating": coating,
        "pitch_mrad": pitch_mrad,
        "cutoff_energy_eV": cutoff,
        "cutoff_definition": "highest energy where s-polarised reflectivity is still 50%",
        "reflectivity_curve": {
            "energy_eV": [float(e) for e in energies[::20]],
            "reflectivity": [float(r) for r in refl[::20]],
        },
    }


def undulator_harmonics(
    period_mm: float, K: float, ring_GeV: float = 3.0, up_to: int = 9
) -> dict:
    """Where this undulator's odd harmonics sit."""
    from .spec import resonance_energy_eV

    return {
        "period_mm": period_mm,
        "K": K,
        "ring_GeV": ring_GeV,
        "harmonics": [
            {"n": n, "energy_eV": resonance_energy_eV(period_mm, K, ring_GeV, n)}
            for n in range(1, up_to + 1, 2)
        ],
    }
