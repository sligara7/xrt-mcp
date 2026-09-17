"""Run the server.

    python -m xrt_mcp                       # stdio, for a local client
    python -m xrt_mcp --transport http      # streamable HTTP, port 8000

Over stdio the transport claims fd 1, so xrt's own chatter cannot corrupt the
protocol. Over HTTP it goes to the console like any other server's logging.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="xrt-mcp", description=__doc__)
    parser.add_argument(
        "--transport", choices=["stdio", "http"], default="stdio",
        help="stdio for a local client (the default), http for remote ones",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    from .server import mcp

    if args.transport == "stdio":
        mcp.run("stdio")
    else:
        mcp.run("streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
