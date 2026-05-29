#!/usr/bin/env python
"""Profile the softbody Franka pipeline (headless, no viewer).

Reports:
  1. Steady-state throughput of the graph-captured frame  (ms/frame, FPS, real-time factor)
  2. Per-kernel CUDA breakdown of one un-captured simulate() (top GPU consumers)

Run:  python playground/profile_franka.py     (uses the shared `sim` conda env)
"""

from __future__ import annotations

import importlib.util
import statistics
import sys
import time
from pathlib import Path

import warp as wp

import newton.examples

# ---- tunables -------------------------------------------------------------
WARMUP_FRAMES = 20
TIMED_FRAMES = 200
TOP_K_KERNELS = 20
# ---------------------------------------------------------------------------

# `softbody/` is not an importable package, so load the example module by path.
_EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "newton/examples/softbody/example_softbody_franka.py"
_spec = importlib.util.spec_from_file_location("example_softbody_franka", _EXAMPLE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Example = _mod.Example


def build_example() -> "Example":
    """Construct the example with a null viewer (no rendering overhead)."""
    sys.argv = ["profile_franka", "--viewer", "null", "--num-frames", str(TIMED_FRAMES)]
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    return Example(viewer, args)


def time_steady_state(ex: "Example") -> None:
    dev = wp.get_device()
    print(f"\n=== steady-state throughput (device={dev}) ===")

    for _ in range(WARMUP_FRAMES):
        ex.step()
    wp.synchronize_device()

    per_frame_ms: list[float] = []
    for _ in range(TIMED_FRAMES):
        t0 = time.perf_counter()
        ex.step()
        wp.synchronize_device()
        per_frame_ms.append((time.perf_counter() - t0) * 1e3)

    mean = statistics.mean(per_frame_ms)
    p50 = statistics.median(per_frame_ms)
    p95 = sorted(per_frame_ms)[int(0.95 * len(per_frame_ms)) - 1]
    fps = 1e3 / mean
    rtf = (ex.frame_dt * 1e3) / mean  # sim-seconds advanced per wall-second

    print(f"  frames timed     : {TIMED_FRAMES}")
    print(f"  ms / frame       : mean {mean:7.3f}   p50 {p50:7.3f}   p95 {p95:7.3f}   min {min(per_frame_ms):7.3f}")
    print(f"  throughput       : {fps:8.1f} frames/s")
    print(f"  sim config       : {ex.fps} fps target, {ex.sim_substeps} substeps "
          f"(sim_dt={ex.sim_dt * 1e3:.3f} ms), VBD iters={ex.iterations}, IK iters={ex.ik_iters}")
    print(f"  real-time factor : {rtf:6.2f}x  ({'faster' if rtf >= 1 else 'SLOWER'} than real time)")


def time_per_kernel(ex: "Example") -> None:
    print(f"\n=== per-kernel CUDA breakdown (one un-captured simulate(), top {TOP_K_KERNELS}) ===")
    if not wp.get_device().is_cuda:
        print("  (CPU device — skipping CUDA kernel timing)")
        return

    # Force the un-captured path so each kernel is individually timeable.
    saved_graph, ex.graph = ex.graph, None
    try:
        with wp.ScopedTimer("simulate", cuda_filter=wp.TIMING_KERNEL, print=False, synchronize=True) as t:
            ex.simulate()
        results = list(getattr(t, "timing_results", []) or [])
    except Exception as e:  # noqa: BLE001 — profiling is best-effort
        print(f"  (cuda_filter timing unavailable: {e})")
        ex.graph = saved_graph
        return
    finally:
        ex.graph = saved_graph

    if not results:
        print("  (no kernel timings captured)")
        return

    agg: dict[str, list[float]] = {}
    for r in results:
        name = getattr(r, "name", "?")
        elapsed = float(getattr(r, "elapsed", 0.0))
        agg.setdefault(name, []).append(elapsed)

    rows = [(name, sum(v), len(v)) for name, v in agg.items()]
    rows.sort(key=lambda x: x[1], reverse=True)
    total = sum(r[1] for r in rows)

    print(f"  total GPU time   : {total:8.3f} ms across {len(results)} launches, {len(rows)} distinct kernels\n")
    print(f"  {'kernel':<52}{'tot ms':>10}{'launches':>10}{'%':>7}")
    print(f"  {'-' * 52}{'-' * 10}{'-' * 10}{'-' * 7}")
    for name, tot, n in rows[:TOP_K_KERNELS]:
        short = name if len(name) <= 50 else name[:47] + "..."
        print(f"  {short:<52}{tot:>10.3f}{n:>10}{100 * tot / total:>6.1f}%")


def main() -> None:
    ex = build_example()
    print(f"model: bodies={ex.model.body_count}  particles={ex.model.particle_count}  "
          f"shapes={ex.model.shape_count}")
    time_steady_state(ex)
    time_per_kernel(ex)


if __name__ == "__main__":
    main()
