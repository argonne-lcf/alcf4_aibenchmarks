#!/usr/bin/env python3
"""Summarize ALCF-4 DBSCAN benchmark runs into the results table.

    python3 DBScan/summarize_results.py <results_dir>/dbscan_n*.out
    python3 DBScan/summarize_results.py --markdown <results_dir>/dbscan_n*.out

Reads the PBS output files written by submit_scaling_aurora.sh, extracts the
per-repetition timings, and prints one row per node count. Runs on a login
node; it only reads text files.

FOM = n * d / T, with d = 3 and T the median of the timed repetitions.

Each job runs 11 repetitions and the first is discarded. Repetition 1 pays
one-off costs the others do not -- paging in a ~1.2 GB executable, allocator
and MPI connection setup -- and including it biases both the median and the
spread. It is still reported, as the "run-1" ratio, because that startup cost
is real; it is typically 1.0-1.5x and does not track node count.

Written for the python3 on the Aurora login nodes (3.6), so no walrus
operator and no statistics.quantiles.
"""
from __future__ import print_function

import argparse
import re
import statistics as s
import sys

DIM = 3


def iqr(xs):
    """Interquartile range, linear interpolation (statistics.quantiles is 3.8+)."""
    xs = sorted(xs)
    n = len(xs)
    if n < 2:
        return 0.0

    def q(p):
        pos = p * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        return xs[lo] + (pos - lo) * (xs[hi] - xs[lo])

    return q(0.75) - q(0.25)


def parse(path):
    """-> (nodes, n_global, [times in repetition order], impl, versions)"""
    nodes = n_global = impl = None
    arborx = kokkos = None
    times = []
    with open(path, errors="replace") as fh:
        for line in fh:
            m = re.search(r"^=== nodes:\s*(\d+)", line)
            if m:
                nodes = int(m.group(1))
                continue
            m = re.search(r"^n \(global\)\s*:\s*(\d+)", line)
            if m:
                n_global = int(m.group(1))
                continue
            m = re.search(r"^total time:\s*([\d.]+)", line)
            if m:
                times.append(float(m.group(1)))
                continue
            m = re.search(r"^implementation\s*:\s*(\S+)", line)
            if m and impl is None:
                impl = m.group(1)
                continue
            m = re.search(r"^ArborX version\s*:\s*(\S+)", line)
            if m and arborx is None:
                arborx = m.group(1)
                continue
            m = re.search(r"^Kokkos version\s*:\s*(\S+)", line)
            if m and kokkos is None:
                kokkos = m.group(1)
    return nodes, n_global, times, impl, (arborx, kokkos)


def main():
    ap = argparse.ArgumentParser(
        description="Summarize DBSCAN benchmark results.")
    ap.add_argument("files", nargs="+", help="PBS output files")
    ap.add_argument("--markdown", action="store_true",
                    help="emit a markdown table for pasting into README.md")
    ap.add_argument("--keep-first", action="store_true",
                    help="include repetition 1 in the statistics (not recommended)")
    args = ap.parse_args()

    runs = {}
    impls = set()
    versions = set()
    for path in args.files:
        nodes, n_global, times, impl, vers = parse(path)
        if nodes is None or not times:
            print("  (skipped %s: no results found)" % path, file=sys.stderr)
            continue
        if nodes in runs:
            print("  (warning: two files report %d nodes; using %s)"
                  % (nodes, path), file=sys.stderr)
        runs[nodes] = (n_global, times)
        if impl:
            impls.add(impl)
        if vers != (None, None):
            versions.add(vers)

    if not runs:
        sys.exit("no parseable results in: %s" % " ".join(args.files))

    if len(impls) > 1:
        print("  (warning: results mix implementations: %s -- these are not"
              " comparable)" % ", ".join(sorted(impls)), file=sys.stderr)
    if len(versions) > 1:
        print("  (warning: results mix library versions: %s)"
              % "; ".join("ArborX %s / Kokkos %s" % v for v in sorted(versions)),
              file=sys.stderr)

    def timed(nodes):
        ts = runs[nodes][1]
        if args.keep_first or len(ts) == 1:
            return ts
        return ts[1:]

    # Weak scaling keeps work-per-rank fixed, so ideal scaling is constant
    # time and efficiency is T(ref)/T(n), not a speedup.
    #
    # The reference is the median time over the smallest quarter of the runs,
    # not the single smallest run. A one-node run is the obvious anchor but is
    # also the least representative point: with one node there is no inter-node
    # communication to overlap and ours came in 10% slower than 2-64 nodes with
    # 3-5x the run-to-run scatter. Anchoring on it alone reported every other
    # point at 106-113% efficiency, which says more about the anchor than the
    # scaling. Taking a median over several small runs removes that sensitivity.
    small = sorted(runs)[:max(1, len(runs) // 4)]
    t_ref = s.median([s.median(timed(n)) for n in small])
    ref_desc = ("%d node%s" % (small[0], "" if small[0] == 1 else "s")
                if len(small) == 1
                else "%d-%d nodes" % (small[0], small[-1]))

    rows = []
    for nodes in sorted(runs):
        g, all_ts = runs[nodes]
        ts = timed(nodes)
        med = s.median(ts)
        mean = s.mean(ts)
        sd = s.stdev(ts) if len(ts) > 1 else 0.0
        rows.append(dict(
            nodes=nodes, n=g, ntimed=len(ts), med=med, mean=mean, sd=sd,
            cv=(sd / mean * 100 if mean else 0.0), iqr=iqr(ts),
            fom=g * DIM / med, eff=t_ref / med * 100,
            r1=(all_ts[0] / mean if len(all_ts) > 1 and not args.keep_first
                else float("nan")),
        ))

    if args.markdown:
        print("| Number of Nodes | n (global) | T (seconds) | FOM | Weak-scaling eff. |")
        print("| -----------: | -----------: | ---: | ---: | ---: |")
        for r in rows:
            print("| %d | %s | %.3f | %s | %.0f%% |"
                  % (r["nodes"], "{:,}".format(r["n"]), r["med"],
                     "{:,.0f}".format(r["fom"]), r["eff"]))
    else:
        hdr = ("%6s %6s %20s %8s %8s %7s %6s %7s %20s %7s %6s"
               % ("nodes", "timed", "n (global)", "T med", "T mean", "sd",
                  "CV%", "IQR", "FOM", "eff", "run-1"))
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            r1 = "    --" if r["r1"] != r["r1"] else "%5.2fx" % r["r1"]
            print("%6d %6d %20s %8.3f %8.3f %7.3f %5.1f%% %7.3f %20s %6.0f%% %6s"
                  % (r["nodes"], r["ntimed"], "{:,}".format(r["n"]), r["med"],
                     r["mean"], r["sd"], r["cv"], r["iqr"],
                     "{:,.0f}".format(r["fom"]), r["eff"], r1))

    print()
    if versions:
        for a, k in sorted(versions):
            print("ArborX %s / Kokkos %s" % (a, k))
    if impls:
        print("implementation: %s" % ", ".join(sorted(impls)))
    print("FOM = n * %d / T, T = median of the timed repetitions." % DIM)
    if not args.keep_first:
        print("Repetition 1 is excluded from the statistics and reported as "
              "'run-1' = t(rep 1) / mean(rest).")
    print("Weak-scaling efficiency is T(ref)/T(n), with T(ref) = %.3f s, the "
          "median over the\nsmallest runs present (%s)." % (t_ref, ref_desc))


if __name__ == "__main__":
    main()
