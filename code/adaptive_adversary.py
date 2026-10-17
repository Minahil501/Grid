"""
GridCopilot — Adaptive adversary evaluation
============================================
    python adaptive_adversary.py RelayMeasurementDataset.csv

Addresses the deepest caveat in the paper: every signature in the knowledge
base detects a PHYSICS VIOLATION, and the benchmark's attacks were generated
by a script that produces such violations. An adversary who knows the detector
would produce injections that are physically coherent. This measures how much
detection depends on attacker incompetence.

METHOD
------
We synthesise attacks at four levels of sophistication, each starting from a
GENUINE fault window so the underlying physics is real, and measure what
fraction the frozen knowledge base still flags as an attack.

  L0  Naive               per-channel random gains plus a hard splice.
                          Reproduces the style of perturbation the benchmark
                          contains, and serves as the control.
  L1  Continuity-aware    the same gains applied through a smooth ramp, so no
                          discontinuity is introduced. Defeats roughness and
                          second-difference signatures only.
  L2  Amplitude-coherent  ONE common gain applied to all six channels,
                          smoothly. Because every descriptor in the knowledge
                          base is a scale-invariant ratio, this manipulation
                          is invisible to all of them by construction.
  L3  Replay              an unmodified genuine fault window presented as the
                          measurement for a different event. No signature can
                          detect this even in principle; it is the upper bound
                          on evasion.

A detection here means the system labelled the synthesised event "attack". An
event labelled "fault" is a FALSE NEGATIVE, which is the dangerous outcome;
abstention is the safe failure.

SPLIT NOTE
----------
Source windows are drawn from the TRAINING split, so no test-split material is
consumed. The quantity measured is evasion of a frozen knowledge base, not
accuracy on held-out labels.
"""

import argparse
import json
import sys

import numpy as np

from tcdr_descriptors import load_tcdr, compute_descriptors, SAMPLES_PER_CYCLE
from calibrate import kb_classify
from knowledge_base import evaluate_signature
from llm_stage import load_frozen_kb

N = SAMPLES_PER_CYCLE


def _smooth_ramp(n=N, start=0.25, width=0.35):
    """A monotone 0->1 ramp with continuous first derivative (raised cosine).

    Using this instead of a step is what separates L1 from L0: the gain change
    is introduced without any second-difference spike, so a discontinuity
    detector sees nothing.
    """
    t = np.arange(n) / n
    r = np.clip((t - start) / width, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * r)


def centred(window):
    med = np.median(window, axis=-1, keepdims=True)
    return window - med, med


def l0_naive(w, rng):
    """Per-channel random gains plus a hard splice. The benchmark's style."""
    x, med = centred(w)
    out = x.copy()
    for c in rng.choice(6, size=rng.integers(2, 5), replace=False):
        out[c] *= rng.uniform(1.25, 1.9)
    k = int(rng.integers(8, 24))
    c = int(rng.integers(0, 6))
    out[c, k:] += rng.uniform(0.15, 0.35) * np.std(out[c])   # step
    return out + med


def l1_continuous(w, rng):
    """Same per-channel gains, introduced through a smooth ramp."""
    x, med = centred(w)
    out = x.copy()
    ramp = _smooth_ramp(start=rng.uniform(0.15, 0.45))
    for c in rng.choice(6, size=rng.integers(2, 5), replace=False):
        g = rng.uniform(1.25, 1.9)
        out[c] = out[c] * (1.0 + (g - 1.0) * ramp)
    return out + med


def l2_coherent(w, rng):
    """ONE gain for all six channels, applied smoothly.

    Every knowledge-base descriptor is a ratio, so a common multiplicative
    factor cancels exactly. This is not an empirical weakness we discovered by
    search; it follows from the scale-invariance that the normalised benchmark
    forced on the descriptor design.
    """
    x, med = centred(w)
    ramp = _smooth_ramp(start=rng.uniform(0.15, 0.45))
    g = rng.uniform(1.3, 2.2)
    return x * (1.0 + (g - 1.0) * ramp) + med


def l3_replay(w, rng, pool):
    """An unmodified genuine fault window from elsewhere in the dataset."""
    return pool[rng.integers(0, len(pool))].copy()


def evaluate(windows, sigs, weights):
    det = fn = ab = 0
    fired = {}
    for w in windows:
        d = compute_descriptors(w).flat()
        y = kb_classify(d, sigs, 0.05, weights)
        if y == "attack":
            det += 1
            for s in sigs:
                if s.event_class == "attack" and evaluate_signature(s, d)["satisfied"]:
                    fired[s.signature_id] = fired.get(s.signature_id, 0) + 1
        elif y == "fault":
            fn += 1
        else:
            ab += 1
    n = len(windows)
    return {"n": n, "detected": det, "false_negative": fn, "abstained": ab,
            "detection_rate": det / n, "false_negative_rate": fn / n,
            "abstention_rate": ab / n, "signatures_fired": fired}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--n", type=int, default=600, help="attacks per level")
    ap.add_argument("--kb", default="calibrated_kb_extended.json")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="adversary_results.json")
    args = ap.parse_args()

    import os
    kb_path = args.kb if os.path.exists(args.kb) else "calibrated_kb_tcdr.json"
    kb = load_frozen_kb(kb_path)
    sigs = list(kb.values())
    wf = ("signature_weights_extended.json"
          if kb_path != "calibrated_kb_tcdr.json" else "signature_weights.json")
    weights = json.load(open(wf)) if os.path.exists(wf) else None
    print(f"Knowledge base: {kb_path}  ({len(sigs)} signatures)")

    W, labels = load_tcdr(args.csv)
    test_idx = set(np.load("tcdr_test_indices.npy").tolist())
    train_faults = [W[i] for i in range(len(W))
                    if i not in test_idx and labels[i] == "fault"]
    print(f"Source material: {len(train_faults)} genuine fault windows "
          f"from the TRAINING split\n")

    rng = np.random.default_rng(args.seed)
    pool = train_faults

    # Control: how does the KB behave on the benchmark's own attacks?
    train_attacks = [W[i] for i in range(len(W))
                     if i not in test_idx and labels[i] == "attack"]
    base = evaluate(train_attacks[:args.n], sigs, weights)
    print(f"{'CONTROL: benchmark attacks':<30} "
          f"detected {base['detection_rate']:.3f}   "
          f"missed {base['false_negative_rate']:.3f}   "
          f"abstained {base['abstention_rate']:.3f}")

    levels = [
        ("L0 naive (benchmark style)", lambda w, r: l0_naive(w, r)),
        ("L1 continuity-aware",        lambda w, r: l1_continuous(w, r)),
        ("L2 amplitude-coherent",      lambda w, r: l2_coherent(w, r)),
        ("L3 replay",                  lambda w, r: l3_replay(w, r, pool)),
    ]

    results = {"control": base, "levels": {}}
    for name, fn in levels:
        src = [pool[i] for i in rng.integers(0, len(pool), args.n)]
        synth = [fn(w, rng) for w in src]
        r = evaluate(synth, sigs, weights)
        results["levels"][name] = r
        print(f"{name:<30} detected {r['detection_rate']:.3f}   "
              f"missed {r['false_negative_rate']:.3f}   "
              f"abstained {r['abstention_rate']:.3f}")

    print("\nWhich attack signatures still fire, by level:")
    for name, r in results["levels"].items():
        f = r["signatures_fired"]
        print(f"  {name:<30} {f if f else 'none'}")

    d0 = results["levels"]["L0 naive (benchmark style)"]["detection_rate"]
    d2 = results["levels"]["L2 amplitude-coherent"]["detection_rate"]
    print("\n--- reading of the result ---")
    print(f"Detection falls from {d0:.3f} against a naive adversary to "
          f"{d2:.3f} against one")
    print("whose manipulation is amplitude-coherent. The knowledge base "
          "detects clumsy")
    print("injection, not injection as such. Report this rather than leave it "
          "for a reviewer.")

    json.dump(results, open(args.out, "w"), indent=2, default=float)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
