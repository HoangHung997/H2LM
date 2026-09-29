from __future__ import annotations

import sys

from .ingest import render_request

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Expected one request JSON path")
    render_request(sys.argv[1])
