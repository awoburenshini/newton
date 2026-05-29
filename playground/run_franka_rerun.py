#!/usr/bin/env python
"""F5-friendly launcher for the softbody Franka example, streaming to a remote Rerun.

Just open this file and press F5 (or `python run_franka_rerun.py`). Everything is
hardcoded below — no command-line args needed. This is equivalent to:

    python -m newton.examples softbody_franka \
        --viewer rerun --rerun-address rerun+http://127.0.0.1:9876/proxy --num-frames 1000

Prereqs for the Rerun stream to show up on your Mac:
  1. On the Mac:   `rerun`                      (viewer listening on :9876)
  2. Reconnect:    `ssh -R 9876:localhost:9876 <this-server>`   (server :9876 -> Mac)
Edit the constants below to change the target, frame count, or viewer.
"""

from __future__ import annotations

import runpy
import sys

# ---- hardcoded run config -------------------------------------------------
EXAMPLE = "softbody_franka"
VIEWER = "rerun"
RERUN_ADDRESS = "rerun+http://127.0.0.1:9876/proxy"
NUM_FRAMES = 1000
# ---------------------------------------------------------------------------

sys.argv = [
    "newton.examples",
    EXAMPLE,
    "--viewer", VIEWER,
    "--rerun-address", RERUN_ADDRESS,
    "--num-frames", str(NUM_FRAMES),
]

# Run `newton.examples` exactly as `python -m newton.examples ...` would.
runpy.run_module("newton.examples", run_name="__main__")
