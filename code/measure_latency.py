"""
GridCopilot — End-to-end latency measurement
=============================================
    python measure_latency.py RelayMeasurementDataset.csv

Closes the "no latency measurement" gap. The paper currently asserts that the
deployed system is "expected to be negligible"; an unmeasured timing claim in a
protection venue is a free objection for a reviewer. This measures it.

WHAT IS TIMED
-------------
The DEPLOYED inference path only, which after Stage 6 contains no LLM:

    raw window -> descriptor extraction -> signature evaluation -> label

Stage 6 (LLM signature authoring) is deliberately NOT timed here: it runs once,
offline, during knowledge-base construction, and never at inference. The
constrained-classifier path of Section V-B is timed separately and reported as
the contrast, since its cost is dominated by a network round trip.

Per-stage timings are reported as median and 95th percentile rather than mean,
because a mean over a few slow outliers misrepresents the typical case, and
protection engineers care about the tail.

No API calls. Runs in about a minute.
"""

import argparse
import json
import platform
import statistics
import sys
import time

import numpy as np

from tcdr_descriptors import load_tcdr, compute_descriptors, SAMPLES_PER_CYCLE
from calibrate import kb_classify
from llm_stage import load_frozen_kb, retrieve

NOMINAL_HZ = 50.0   # one AC cycle per 32-sample window; change to 60 if needed


def timed(fn, reps):
    """Return per-call times in milliseconds."""
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def stats(xs):
    xs = sorted(xs)
    return {
        "median_ms": statistics.median(xs),
        "mean_ms": statistics.fmean(xs),
        "p95_ms": xs[int(0.95 * (len(xs) - 1))],
        "max_ms": xs[-1],
        "n": len(xs),
    }


def show(name, s, cycle_ms):
    print(f"{name:<34} median {s['median_ms']:>7.3f} ms   "
          f"p95 {s['p95_ms']:>7.3f} ms   "
          f"({s['median_ms']/cycle_ms:>5.2f} cycles)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--n", type=int, default=2000, help="events to time")
    ap.add_argument("--kb", default="calibrated_kb_extended.json",
                    help="knowledge base to time (falls back to the base KB)")
    ap.add_argument("--hz", type=float, default=NOMINAL_HZ)
    ap.add_argument("--out", default="latency_results.json")
    args = ap.parse_args()

    cycle_ms = 1000.0 / args.hz
    print(f"Nominal system frequency {args.hz:g} Hz -> one AC cycle = "
          f"{cycle_ms:.2f} ms\n")

    import os
    kb_path = args.kb if os.path.exists(args.kb) else "calibrated_kb_tcdr.json"
    if kb_path != args.kb:
        print(f"({args.kb} not found; timing {kb_path} instead)\n")
    kb = load_frozen_kb(kb_path)
    sigs = list(kb.values())
    try:
        weights = json.load(open("signature_weights_extended.json"
                                 if kb_path != "calibrated_kb_tcdr.json"
                                 else "signature_weights.json"))
    except FileNotFoundError:
        weights = None

    W, labels = load_tcdr(args.csv)
    idx = np.arange(min(args.n, len(W)))
    print(f"Timing {len(idx)} events against {len(sigs)} signatures\n")

    # --- Stage 1: descriptor extraction ---
    t_desc = timed(lambda: [compute_descriptors(W[i]) for i in idx[:1]], 1)
    t_desc = []
    for i in idx:
        t0 = time.perf_counter()
        d = compute_descriptors(W[i])
        t_desc.append((time.perf_counter() - t0) * 1000.0)
    flats = [compute_descriptors(W[i]).flat() for i in idx]

    # --- Stage 2+5: signature evaluation and voting ---
    t_cls = []
    for f in flats:
        t0 = time.perf_counter()
        kb_classify(f, sigs, 0.05, weights)
        t_cls.append((time.perf_counter() - t0) * 1000.0)

    # --- Retrieval (used only by the constrained-classifier variant) ---
    t_ret = []
    for f in flats[:500]:
        t0 = time.perf_counter()
        retrieve(f, kb, 5)
        t_ret.append((time.perf_counter() - t0) * 1000.0)

    # --- End to end, deployed path ---
    t_e2e = []
    for i in idx:
        t0 = time.perf_counter()
        f = compute_descriptors(W[i]).flat()
        kb_classify(f, sigs, 0.05, weights)
        t_e2e.append((time.perf_counter() - t0) * 1000.0)

    s_desc, s_cls, s_ret, s_e2e = (stats(t_desc), stats(t_cls),
                                   stats(t_ret), stats(t_e2e))

    print("DEPLOYED PATH (no LLM at inference)")
    show("  Stage 1 descriptor extraction", s_desc, cycle_ms)
    show("  Signature evaluation + vote", s_cls, cycle_ms)
    show("  END TO END", s_e2e, cycle_ms)
    print()
    print("CONSTRAINED-CLASSIFIER VARIANT (Section V-B), local portion only")
    show("  Signature retrieval", s_ret, cycle_ms)
    print("  + one LLM API round trip, network-bound, measured separately")
    print("    from the run logs; this is the component that makes the")
    print("    classifier variant unsuitable for the protection path.")

    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "numpy": np.__version__,
    }
    print("\nEnvironment")
    for k, v in env.items():
        print(f"  {k}: {v}")

    sentence = (
        f"On {env['platform'].split('-')[0]} with Python {env['python']}, the "
        f"deployed path requires a median of {s_e2e['median_ms']:.3f} ms per "
        f"event ({s_e2e['p95_ms']:.3f} ms at the 95th percentile), of which "
        f"{s_desc['median_ms']:.3f} ms is descriptor extraction and "
        f"{s_cls['median_ms']:.3f} ms is signature evaluation over "
        f"{len(sigs)} signatures. At {args.hz:g} Hz this is "
        f"{s_e2e['median_ms']/cycle_ms:.3f} AC cycles."
    )
    print("\n--- paste-ready sentence for Section V ---\n" + sentence)

    json.dump({"cycle_ms": cycle_ms, "n_signatures": len(sigs),
               "descriptor": s_desc, "classify": s_cls, "retrieval": s_ret,
               "end_to_end": s_e2e, "environment": env,
               "sentence": sentence},
              open(args.out, "w"), indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
