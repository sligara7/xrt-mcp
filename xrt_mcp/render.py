"""Turning a traced beam into a picture.

Headless, always: the Agg backend is selected on import, before pyplot is
touched, because an MCP server has no display and matplotlib's default backend
would try to find one.

Everything here returns PNG bytes. Nothing here writes a file.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .tracing import ScreenReading, TraceResult  # noqa: E402

_BG = "#ffffff"
_FG = "#1f2328"
_MUTED = "#6b7280"


def _finish(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    return buf.getvalue()


def footprint(reading: ScreenReading, bins: int | None = None) -> bytes:
    """The beam where it lands: an intensity-weighted image, with the profiles
    through it and the numbers that matter written on."""
    if reading._x_mm is None or reading.rays_alive == 0:
        return _empty(f"no rays reached {reading.name}")

    # bin count follows the ray count: 120 bins over 500 rays is a picture of
    # the sampling noise, not of the beam
    if bins is None:
        bins = int(min(120, max(24, (reading.rays_alive ** 0.5) / 2)))

    x_um = reading._x_mm * 1000.0
    z_um = reading._z_mm * 1000.0
    w = reading._I

    fig = plt.figure(figsize=(9.0, 4.0), facecolor=_BG)
    grid = fig.add_gridspec(2, 3, width_ratios=[3, 1, 2.4], height_ratios=[1, 3],
                            hspace=0.06, wspace=0.28)
    ax = fig.add_subplot(grid[1, 0])
    ax_top = fig.add_subplot(grid[0, 0], sharex=ax)
    ax_right = fig.add_subplot(grid[1, 1], sharey=ax)
    ax_text = fig.add_subplot(grid[:, 2])

    h, xe, ze = np.histogram2d(x_um, z_um, bins=bins, weights=w)
    ax.pcolormesh(xe, ze, h.T, cmap="inferno", shading="auto")
    ax.set_xlabel("horizontal [µm]", color=_FG)
    ax.set_ylabel("vertical [µm]", color=_FG)

    ax_top.hist(x_um, bins=bins, weights=w, color="#c2410c", histtype="stepfilled")
    ax_right.hist(z_um, bins=bins, weights=w, color="#c2410c",
                  histtype="stepfilled", orientation="horizontal")
    for a in (ax_top, ax_right):
        a.axis("off")

    ax_text.axis("off")
    lines = [
        f"{reading.name}   at {reading.at_m:g} m",
        "",
        f"FWHM   {_fmt(reading.x_fwhm_um)} × {_fmt(reading.z_fwhm_um)} µm",
        f"rms    {_fmt(reading.x_rms_um)} × {_fmt(reading.z_rms_um)} µm",
        "",
        f"energy {_fmt(reading.mean_energy_eV, 1)} eV",
        f"ΔE     {_fmt(reading.energy_fwhm_eV, 2)} eV FWHM",
        "",
        f"transmitted {reading.transmission * 100:.3g} % of source",
    ]
    if reading.flux_ph_s is not None:
        lines.append(f"flux   {reading.flux_ph_s:.3g} ph/s")
    if reading.power_W is not None:
        lines.append(f"power  {reading.power_W:.3g} W")
    lines.append("")
    lines.append(f"{reading.rays_alive} rays")
    ax_text.text(0, 1, "\n".join(lines), va="top", ha="left", family="monospace",
                 fontsize=9, color=_FG, transform=ax_text.transAxes)

    fig.patch.set_facecolor(_BG)
    return _finish(fig)


def spectrum(reading: ScreenReading, bins: int | None = None) -> bytes:
    """What energies actually got through."""
    if reading._E_eV is None or reading.rays_alive == 0:
        return _empty(f"no rays reached {reading.name}")
    if bins is None:
        bins = int(min(120, max(20, (reading.rays_alive ** 0.5) / 2)))
    fig, ax = plt.subplots(figsize=(7.0, 3.2), facecolor=_BG)
    ax.hist(reading._E_eV, bins=bins, weights=reading._I,
            color="#1d4ed8", histtype="stepfilled")
    ax.set_xlabel("photon energy [eV]", color=_FG)
    ax.set_ylabel("intensity [arb.]", color=_FG)
    ax.set_title(f"{reading.name} at {reading.at_m:g} m — "
                 f"ΔE = {_fmt(reading.energy_fwhm_eV, 2)} eV FWHM",
                 color=_FG, fontsize=10)
    return _finish(fig)


def layout(result: TraceResult, spec) -> bytes:
    """Where the beam is along the beamline — spot size against distance, so a
    focus is visible as a waist rather than as a table row."""
    xs = [r.at_m for r in result.readings]
    fig, ax = plt.subplots(figsize=(7.0, 3.2), facecolor=_BG)
    for key, colour, label in (("x_fwhm_um", "#c2410c", "horizontal"),
                               ("z_fwhm_um", "#1d4ed8", "vertical")):
        ys = [getattr(r, key) for r in result.readings]
        ax.plot(xs, ys, "o-", color=colour, label=label)
    for r in result.readings:
        ax.annotate(r.name, (r.at_m, r.x_fwhm_um or 0), fontsize=8,
                    color=_MUTED, xytext=(0, 8), textcoords="offset points",
                    ha="center")
    ax.set_yscale("log")
    ax.set_xlabel("distance from source [m]", color=_FG)
    ax.set_ylabel("beam size, FWHM [µm]", color=_FG)
    ax.set_title(result.beamline, color=_FG, fontsize=10)
    ax.legend(frameon=False)
    return _finish(fig)


def scan_curve(x_values, y_values, x_label: str, y_label: str, title: str) -> bytes:
    fig, ax = plt.subplots(figsize=(7.0, 3.2), facecolor=_BG)
    ax.plot(x_values, y_values, "o-", color="#047857")
    ax.set_xlabel(x_label, color=_FG)
    ax.set_ylabel(y_label, color=_FG)
    ax.set_title(title, color=_FG, fontsize=10)
    return _finish(fig)


def _empty(message: str) -> bytes:
    fig, ax = plt.subplots(figsize=(5.0, 1.6), facecolor=_BG)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", color=_MUTED, fontsize=11)
    return _finish(fig)


def _fmt(value, places: int = 1) -> str:
    return "—" if value is None else f"{value:.{places}f}"
