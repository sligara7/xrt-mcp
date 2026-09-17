"""What every answer carries about itself.

A number from this server may be written into a design, quoted months later, or
compared against the same calculation on another machine. When two machines
disagree, the question is always the same: is this a different model, or a
different build? Nothing in the answer could tell you, so this module exists.

The split is deliberate. `Provenance` is three fields and rides on every reply
that carries a number, because a version is only useful attached to the thing it
produced and a tool you have to remember to call is a tool that gets forgotten.
Everything else — Python, platform, the numeric stack — is a `server_info` call
away, because it is the same for every reply in a session and paying for it on
each one would buy nothing.
"""

from __future__ import annotations

import datetime as dt
import platform
import sys
from functools import lru_cache

from pydantic import BaseModel, Field

from . import __version__


@lru_cache(maxsize=1)
def xrt_version() -> str:
    try:
        from xrt.version import __version__ as v

        return str(v)
    except Exception:  # pragma: no cover - xrt is a hard dependency
        return "unknown"


class Provenance(BaseModel):
    """Which build computed this, and when. Rides on every answer."""

    xrt_mcp: str = Field(description="version of this server")
    xrt: str = Field(description="version of the xrt library that did the physics")
    computed_at: str = Field(
        description="when the server computed it, UTC — not when you read it"
    )


def provenance() -> Provenance:
    return Provenance(
        xrt_mcp=__version__,
        xrt=xrt_version(),
        computed_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    )


class ServerInfo(BaseModel):
    """The whole environment, for the once-per-session question."""

    xrt_mcp: str
    xrt: str
    python: str
    platform: str
    numpy: str
    scipy: str
    matplotlib: str
    mcp_sdk: str = Field(
        description="which SDK generation; 2.x is MCPServer, 1.x was FastMCP"
    )


def server_info() -> ServerInfo:
    def _v(module: str) -> str:
        try:
            import importlib

            return str(getattr(importlib.import_module(module), "__version__", "unknown"))
        except Exception:
            return "unknown"

    return ServerInfo(
        xrt_mcp=__version__,
        xrt=xrt_version(),
        python=sys.version.split()[0],
        platform=f"{platform.system()} {platform.release()} {platform.machine()}",
        numpy=_v("numpy"),
        scipy=_v("scipy"),
        matplotlib=_v("matplotlib"),
        mcp_sdk=_v("mcp") if _v("mcp") != "unknown" else "2.x",
    )
