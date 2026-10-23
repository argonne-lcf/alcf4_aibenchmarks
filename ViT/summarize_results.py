#!/usr/bin/env python3
"""Summarize ALCF-4 3D ViT benchmark logs into the FOM tables.

Usage:
    python3 summarize_results.py [results_dir] [--warmup N] [--csv out.csv]

Reads the logs written by run_vit_aurora.sh (vit_n<N>_factor<F>_sp<S>_rep<R>.log)
and reports, per configuration, the steady-state iteration time and the
achieved FLOP rate that the figure of merit is defined from.

Each training iteration prints a line like:

    iteration:7 \t time:5.25 \t TFLOPS:771.34 \t TFLOPS_per_gpu:64.28 \t ...

The first iterations include warmup -- JIT compilation, allocator growth,
collective setup -- and are not representative. On Aurora the rate is flat
from about iteration 6 onward, so the default discards the first 5. The
reported spread makes it visible when that assumption does not hold for a
configuration.
"""
import argparse
import csv
import math
import os
import re
import sys
from collections import defaultdict

# iteration:7 time:5.25 TFLOPS:771.34 TFLOPS_per_gpu:64.28 Samples/Sec:2.3 ...
ITER_RE = re.compile(
    r"iteration:(\d+)\s+time:([\d.]+)\s+TFLOPS:([\d.]+)\s+"
    r"TFLOPS_per_gpu:([\d.]+)\s+Samples/Sec:([\d.]+)"
)
# vit_n128_factor32_sp1_rep1.log
NAME_RE = re.compile(r"vit_n(\d+)_factor(\d+)_sp(\d+)_rep(\d+)\.log$")

RANKS_PER_NODE = 12


def parse_log(path):
    """Return (iteration, time, tflops, tflops_per_gpu, samples_per_sec) rows."""
    rows = []
    with open(path, errors="replace") as fh:
        for line in fh:
            m = ITER_RE.search(line)
            if m:
                rows.append((int(m.group(1)), float(m.group(2)),
                             float(m.group(3)), float(m.group(4)),
                             float(m.group(5))))
    return rows


def stats(values):
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
        sd = math.sqrt(var)
    else:
        sd = 0.0
    return mean, sd, min(values), max(values)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", nargs="?", default=None,
                    help="directory of logs (default: ViT/results next to this script)")
    ap.add_argument("--warmup", type=int, default=5,
                    help="leading iterations to discard (default: 5)")
    ap.add_argument("--csv", default=None, help="also write a CSV here")
    args = ap.parse_args()

    results_dir = args.results_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results")
    if not os.path.isdir(results_dir):
        sys.exit(f"no such directory: {results_dir}")

    logs = sorted(f for f in os.listdir(results_dir) if NAME_RE.search(f))
    if not logs:
        sys.exit(f"no vit_n*_factor*_sp*_rep*.log files in {results_dir}")

    # (nodes, factor, sp) -> list of per-repetition summaries
    runs = defaultdict(list)
    incomplete = []

    for name in logs:
        m = NAME_RE.search(name)
        nodes, factor, sp, rep = (int(m.group(i)) for i in (1, 2, 3, 4))
        rows = parse_log(os.path.join(results_dir, name))
        kept = [r for r in rows if r[0] > args.warmup]
        if not kept:
            incomplete.append((name, len(rows)))
            continue
        t_mean, t_sd, t_min, t_max = stats([r[1] for r in kept])
        fl_mean = sum(r[2] for r in kept) / len(kept)
        sps_mean = sum(r[4] for r in kept) / len(kept)
        runs[(nodes, factor, sp)].append({
            "rep": rep, "n_iter": len(kept), "t_mean": t_mean, "t_sd": t_sd,
            "t_min": t_min, "t_max": t_max, "tflops": fl_mean, "sps": sps_mean,
        })

    if incomplete:
        print("Logs with no timed iterations after warmup "
              f"(discarding first {args.warmup}):")
        for name, n in incomplete:
            print(f"  {name}  ({n} iteration line(s) found)")
        print()

    # ---- per-configuration detail -------------------------------------
    print(f"Steady-state per configuration (first {args.warmup} iterations discarded)")
    print()
    hdr = (f"{'nodes':>6} {'factor':>7} {'tokens':>10} {'SP':>4} {'reps':>5} "
           f"{'s/iter':>9} {'spread':>8} {'TFLOPS':>10} {'TFLOP/tile':>11} {'samp/s':>8}")
    print(hdr)
    print("-" * len(hdr))

    csv_rows = []
    for (nodes, factor, sp) in sorted(runs):
        entries = runs[(nodes, factor, sp)]
        # Across repetitions, take the mean of the per-rep means; report the
        # worst-case spread seen within any single repetition.
        t = sum(e["t_mean"] for e in entries) / len(entries)
        fl = sum(e["tflops"] for e in entries) / len(entries)
        sps = sum(e["sps"] for e in entries) / len(entries)
        spread = max((e["t_max"] - e["t_min"]) / e["t_mean"] for e in entries)
        tiles = nodes * RANKS_PER_NODE
        print(f"{nodes:>6} {factor:>7} {factor**3:>10,} {sp:>4} {len(entries):>5} "
              f"{t:>9.3f} {spread*100:>7.1f}% {fl:>10.1f} {fl/tiles:>11.2f} {sps:>8.2f}")
        csv_rows.append({
            "nodes": nodes, "factor": factor, "tokens": factor**3, "sp": sp,
            "reps": len(entries), "sec_per_iter": round(t, 4),
            "spread_pct": round(spread * 100, 2), "tflops": round(fl, 2),
            "tflops_per_tile": round(fl / tiles, 3), "samples_per_sec": round(sps, 3),
        })

    # ---- weak-scaling view, per sequence length -----------------------
    by_factor = defaultdict(dict)
    for (nodes, factor, sp) in runs:
        entries = runs[(nodes, factor, sp)]
        by_factor[factor][nodes] = sum(e["tflops"] for e in entries) / len(entries)

    for factor in sorted(by_factor):
        series = by_factor[factor]
        if len(series) < 2:
            continue
        base_nodes = min(series)
        base = series[base_nodes] / base_nodes
        print()
        print(f"Weak scaling, factor={factor} ({factor**3:,} tokens), "
              f"baseline {base_nodes} node(s)")
        print(f"{'nodes':>6} {'TFLOPS':>12} {'TFLOPS/node':>12} {'efficiency':>11}")
        print("-" * 43)
        for nodes in sorted(series):
            fl = series[nodes]
            print(f"{nodes:>6} {fl:>12.1f} {fl/nodes:>12.2f} "
                  f"{100*(fl/nodes)/base:>10.0f}%")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
