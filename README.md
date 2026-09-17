# xrt-mcp

The [xrt](https://xrt.readthedocs.io) X-ray ray-tracing library behind the Model
Context Protocol, so that a language model can model a synchrotron beamline:
tune an undulator, pick a monochromator, focus with a mirror, and read the flux
and the spot size at the sample.

This is a **demo**, built to answer one question — *would a beamline group want
this?* — not a finished product. What it does, it does properly; what it does
not do is listed at the bottom, honestly.

## What it can do

Nine tools. Four of them answer optics questions instantly, without tracing
anything:

| tool | question it answers |
|---|---|
| `crystal_at_energy` | Bragg angle, Darwin width and intrinsic bandwidth of Si(111)/(220)/(311)/(333) |
| `crystal_energy_range` | what energies a crystal can reach at all |
| `mirror_cutoff` | the critical energy of a coating at a grazing angle, and its reflectivity curve |
| `undulator_harmonics` | where the odd harmonics of a given gap sit |

and five that work on a whole beamline:

| tool | what it does |
|---|---|
| `list_beamlines` | the ready-made beamlines to start from |
| `load_beamline` | one of them, as an editable spec, tuned to an energy |
| `trace_beamline` | propagates rays; returns beam size, bandwidth, transmission, flux, power |
| `beam_image` | the same, as a picture — footprint, spectrum, or beam size along the beamline |
| `scan_element` | sweeps one number (slit gap, mono energy, mirror pitch) and returns the curve |

## What a session looks like

Load the NSLS-II-like beamline at 9 keV, trace it, and this comes back:

```
screen              at    H FWHM    V FWHM       dE    transm
-------------------------------------------------------------
front_end          25m    1723.6     561.0    49.99      100%
after_dcm          31m    1178.7     722.1     1.56     0.99%
sample             45m      22.6       2.0     1.56    0.925%
```

A millimetre-scale white beam with a 50 eV window arrives at the front end; the
Si(111) monochromator cuts it to 1.56 eV; the toroidal mirror puts it into
23 × 2 µm at the sample. Independently, `crystal_at_energy` predicts a 1.23 eV
intrinsic bandwidth for Si(111) at 9 keV — the trace is wider because the
incident beam has divergence, and divergence maps onto energy through
cot θ_B. Two different paths through the physics, agreeing. There is a test
that asserts exactly this (`tests/test_undulator_physics.py`).

With a real undulator source, `flux_ph_s` and `power_W` columns appear as well.

## Why a spec and not a script

An agent sends a beamline as **data** — a source, a list of elements — and this
server builds the xrt objects itself. It never executes Python that an agent
wrote or chose.

That was a deliberate choice and it cost something: anything expressible in xrt
but not in the spec is unreachable until the spec grows. It bought two things.
The obvious one is that an HTTP-reachable server that execs supplied Python is
arbitrary code execution on whatever machine xrt is installed on. The less
obvious one is that a typed spec is something a model can reason about — it can
see that a `Mirror` has a `pitch_mrad` and a `focus_at_m` — where writing xrt
code blind is guesswork.

## How it is put together

```
xrt_mcp/
  server.py      the MCP surface: tools, and nothing else
  spec.py        the beamline description, and the builder that makes xrt objects
  templates.py   ready-made beamlines
  tracing.py     runs the trace, reduces it to numbers
  render.py      turns a traced beam into a PNG
  reference.py   optics questions answered without tracing
```

Five small modules rather than one big one because xrt 2.0 is a beta that is
still moving, and each of these mirrors one part of xrt's own shape
(`sources` / `oes` / `materials` / `plotter`). When xrt changes, the change
should land in one file.

### Geometry

Positions are given as distance along the beam from the source, in metres. The
builder tracks where the beam actually **is** — a monochromator lifts it by its
fixed offset, a mirror deflects it by twice its grazing angle — and places each
element on the real beam path rather than on a nominal axis. That is why a
beamline with both a DCM and a deflecting mirror stays centred on every screen
downstream instead of walking off them.

### Flux and power

xrt gives each ray of a synchrotron source a weight such that its documented
estimator holds:

```
flux  = Σ Tᵢ wᵢ            photons/second
power = Σ Tᵢ Eᵢ e wᵢ       watts
```

for any downstream transmission factor `Tᵢ`, which after propagation is just
the ray's surviving intensity. These numbers are xrt's, not a re-derivation.

Two caveats, and they matter: they are the flux **inside the sampled window**
— the source's energy range and angular half-acceptance — so widening
`energy_min_eV`/`energy_max_eV` genuinely changes the answer. And a geometric
source carries no such weight, so flux comes back as `null` for one rather than
as a number that would look real.

## Running it

Both dependencies are local checkouts, not PyPI releases: xrt 2.0.0b1 and the
mcp Python SDK 2.x. (This is mcp **2.x** — `MCPServer`, not `FastMCP`. Almost
every MCP example in circulation is v1 code and will not import here.)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy scipy matplotlib colorama distro
uv pip install --python .venv/bin/python -e /path/to/python-sdk
uv pip install --python .venv/bin/python --no-deps -e /path/to/xrt
```

`--no-deps` on xrt is deliberate: xrt declares pyopencl, pyopengl and a GUI
stack that the ray-tracing path here does not touch.

Then, as a local MCP server over stdio:

```bash
.venv/bin/python -m xrt_mcp
```

or over streamable HTTP:

```bash
.venv/bin/python -m xrt_mcp --transport http --port 8000
```

To register it with Claude Code:

```bash
claude mcp add xrt -- /path/to/xrt-mcp/.venv/bin/python -m xrt_mcp
```

### The demo

```bash
.venv/bin/python demo/run_demo.py          # real undulator emission, ~2 minutes
.venv/bin/python demo/run_demo.py --fast   # geometric source, ~15 seconds
```

It starts the server the way a client would, over stdio, and walks through the
questions a beamline scientist would actually ask. Images land in
`demo/output/`. Every call it makes is one a model can make on its own; the
script only fixes the order.

### Tests

```bash
.venv/bin/python -m pytest          # 6 end-to-end tests over the real protocol
.venv/bin/python -m pytest -m slow  # 4 more, with real undulator emission
```

The end-to-end tests drive a genuine MCP client against the server over stdio,
so a tool whose signature the SDK cannot serialise fails in the test rather
than in front of somebody.

## What it does not do

Being a demo, the boundaries are real and worth stating before anyone leans on
it:

- **Four element types**: apertures, a DCM, mirrors (flat or toroidal), and
  screens. No gratings, no CRLs, no multilayers, no crystal analysers, no
  bending-magnet or wiggler sources — xrt has all of them; the spec does not
  reach them yet.
- **Ray tracing only.** xrt's wave-propagation and coherence machinery is
  untouched, and that is a large part of why people use xrt.
- **The templates are plausible, not as-built.** They are shaped like NSLS-II
  hard x-ray beamlines — 3 GeV, 500 mA, in-vacuum undulator, Si(111) DCM — but
  no distance in them came from a real beamline drawing.
- **Stateless.** Every call carries the whole beamline, and every call re-traces
  from scratch. An undulator trace is 20–60 seconds, so a ten-point scan on a
  real source is ten minutes. Caching a built beamline between calls is the
  obvious next thing.
- **No authentication on the HTTP transport.** Fine on localhost; not something
  to expose.
- **Vertical deflection only** for mirrors, and the horizontal plane is left
  alone.
