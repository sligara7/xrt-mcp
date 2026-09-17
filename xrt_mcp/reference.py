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


def mirror_cutoff(
    coating: str, pitch_mrad: float, include_curve: bool = False
) -> dict:
    """The critical energy of a mirror at a grazing angle — above it the mirror
    stops reflecting, which is exactly how a mirror is used to kill harmonics.

    Also reports the coating's absorption edges inside the range. Those matter
    for the same job: an edge drops the reflectivity in a band well BELOW the
    cutoff, and it does not come back afterwards. Rh at 3 mrad is the standard
    example — the three L edges between 3.0 and 3.4 keV take it from 0.97 to
    0.88, and it is still only 0.91 at 4.2 keV.

    The curve is off by default. When it is on, it is returned at the full
    resolution it was computed at and NOT decimated: sampling a reflectivity
    curve more coarsely than its edges is how a step gets drawn as a notch.
    """
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

    out = {
        "coating": coating,
        "pitch_mrad": pitch_mrad,
        "cutoff_energy_eV": cutoff,
        "cutoff_definition": "highest energy where s-polarised reflectivity is still 50%",
        "absorption_edges": _absorption_edges(energies, refl, cutoff),
        "edge_definition": (
            "a step down that the local trend of the curve does not explain, found "
            "below the cutoff only — above it the reflectivity is falling steeply "
            "anyway. Energies are the top of the step, accurate to the grid spacing, "
            "about 1%"
        ),
    }
    if include_curve:
        out["reflectivity_curve"] = {
            "energy_eV": [float(e) for e in energies],
            "reflectivity": [float(r) for r in refl],
            "points": int(energies.size),
            "note": "full computed resolution, not decimated",
        }
    return out


def _absorption_edges(
    energies: np.ndarray, refl: np.ndarray, cutoff: float | None
) -> list[dict]:
    """Find the steps down in a reflectivity curve that are absorption edges.

    Two things make this harder than thresholding a drop, and both were got
    wrong on the first attempt. Above the critical angle the reflectivity is
    falling steeply anyway, so any fixed threshold reports the entire roll-off
    as edges — 110 of them, for Rh. And the interesting region is precisely the
    one BELOW the cutoff, where the mirror is supposed to be working and an edge
    is an unwelcome surprise.

    So: look only below the cutoff, and in log-log space, where the smooth part
    of the curve is nearly a straight line. An edge is then a drop the local
    trend does not explain, measured against a robust spread of the residuals
    rather than an absolute number, so it works for any coating and angle.

    Detected from the curve rather than read from an edge table on purpose: it
    reports the edges actually in THIS answer, for whatever coating xrt was
    asked about, instead of a table that has to be kept in step with xrt's own
    atomic data.
    """
    if cutoff is None:
        return []
    keep = energies <= cutoff
    e, r = energies[keep], refl[keep]
    if e.size < 32:
        return []

    slope = np.diff(np.log(np.maximum(r, 1e-12))) / np.diff(np.log(e))
    window = 15
    padded = np.pad(slope, window // 2, mode="edge")
    baseline = np.array(
        [np.median(padded[i : i + window]) for i in range(slope.size)]
    )
    resid = slope - baseline
    spread = np.median(np.abs(resid - np.median(resid)))
    if spread <= 0:
        return []
    hits = np.nonzero(resid < -8.0 * spread)[0]

    # adjacent samples belong to one edge; report the whole step, not each sample
    edges: list[dict] = []
    for group in np.split(hits, np.nonzero(np.diff(hits) > 2)[0] + 1):
        if group.size == 0:
            continue
        lo, hi = int(group[0]), int(group[-1]) + 1
        if r[lo] - r[hi] < 0.005:  # too shallow to matter to anyone
            continue
        # the group spans the whole shoulder; the edge itself is the steepest
        # single drop inside it, and it lies between that pair of samples
        steepest = lo + int(np.argmin(np.diff(r[lo : hi + 1])))
        edges.append(
            {
                "energy_eV": float(np.sqrt(e[steepest] * e[steepest + 1])),
                "reflectivity_below": float(r[lo]),
                "reflectivity_above": float(r[hi]),
            }
        )
    return edges


def undulator_harmonics(
    period_mm: float, K: float, ring_GeV: float, up_to: int = 9
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
