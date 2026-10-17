"""
GridCopilot — Evaluate the extended knowledge base
===================================================
    python eval_extended.py RelayMeasurementDataset.csv

Opens the sealed test split ONCE and compares:
    original KB (hand-written)   vs   extended KB (+ LLM-proposed signatures)

This is the number that decides whether the LLM contributed. Run it after
propose_signatures.py, and do not go back and propose more after seeing the
result -- that would be tuning on test, and every figure in the paper would
become unreportable.
"""

import json
import sys
import numpy as np
from sklearn.metrics import classification_report, balanced_accuracy_score

from tcdr_descriptors import load_tcdr, compute_descriptors
from knowledge_base import Signature
from calibrate import kb_classify
from compare_systems import wilson


def load_kb(path):
    return [Signature(**r) for r in json.load(open(path))]


def evaluate(name, sigs, weights, descs, truth, margin=0.05):
    preds = [kb_classify(d, sigs, margin, weights) for d in descs]
    n = len(truth)
    cov = [(p, t) for p, t in zip(preds, truth) if p is not None]
    print(f"\n--- {name} ({len(sigs)} signatures) ---")
    print(f"coverage: {len(cov)/n:.4f}  ({len(cov)}/{n})")
    if cov:
        yp = [p for p, _ in cov]; yt = [t for _, t in cov]
        k = sum(p == t for p, t in cov)
        lo, hi = wilson(k, len(cov))
        print(f"selective accuracy: {k/len(cov):.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
        print(f"selective balanced acc: {balanced_accuracy_score(yt, yp):.4f}")
        print(classification_report(yt, yp, digits=4, zero_division=0))
    tp = sum(1 for p, t in zip(preds, truth) if p == "attack" and t == "attack")
    fp = sum(1 for p, t in zip(preds, truth) if p == "attack" and t == "fault")
    n_atk = sum(1 for t in truth if t == "attack")
    lo, hi = wilson(tp, n_atk)
    print(f"attack recall over ALL events: {tp}/{n_atk} = {tp/max(n_atk,1):.4f}"
          f"  95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"false attack alarms: {fp}")
    return preds, {"coverage": len(cov)/n,
                   "selective_acc": (sum(p == t for p, t in cov)/len(cov)) if cov else 0,
                   "attack_recall": tp/max(n_atk, 1), "false_alarms": fp}


def main(csv_path):
    W, labels = load_tcdr(csv_path)
    test_idx = np.load("tcdr_test_indices.npy")
    print(f"Sealed test split: {len(test_idx)} events")
    descs = [compute_descriptors(W[i]).flat() for i in test_idx]
    truth = [labels[i] for i in test_idx]
    print(f"  {truth.count('attack')} attack, {truth.count('fault')} fault")

    base = load_kb("calibrated_kb_tcdr.json")
    w_base = json.load(open("signature_weights.json"))
    ext = load_kb("calibrated_kb_extended.json")
    w_ext = json.load(open("signature_weights_extended.json"))

    p_base, m_base = evaluate("ORIGINAL KB (hand-written)", base, w_base, descs, truth)
    p_ext, m_ext = evaluate("EXTENDED KB (+ LLM-proposed)", ext, w_ext, descs, truth)

    print("\n===== DID THE LLM CONTRIBUTE? =====")
    for k, label in [("coverage", "coverage"),
                     ("attack_recall", "attack recall"),
                     ("selective_acc", "selective accuracy")]:
        d = m_ext[k] - m_base[k]
        arrow = "+" if d > 0 else ""
        print(f"{label:<20} {m_base[k]:.4f} -> {m_ext[k]:.4f}   ({arrow}{d:.4f})")
    print(f"{'false alarms':<20} {m_base['false_alarms']} -> "
          f"{m_ext['false_alarms']}")

    rescued = [(i, t) for i, b, e, t in zip(test_idx, p_base, p_ext, truth)
               if b is None and e is not None]
    broke = [(i, t) for i, b, e, t in zip(test_idx, p_base, p_ext, truth)
             if b is not None and e is None]
    flipped_right = [(i, t) for i, b, e, t in zip(test_idx, p_base, p_ext, truth)
                     if b is not None and e is not None and b != t and e == t]
    flipped_wrong = [(i, t) for i, b, e, t in zip(test_idx, p_base, p_ext, truth)
                     if b is not None and e is not None and b == t and e != t]
    print(f"\nnewly covered (was abstain): {len(rescued)}  "
          f"({sum(1 for _, t in rescued if t == 'attack')} attacks)")
    print(f"corrected (was wrong, now right): {len(flipped_right)}")
    print(f"broken (was right, now wrong):    {len(flipped_wrong)}")
    print(f"lost coverage:                    {len(broke)}")

    n_new = len(ext) - len(base)
    if m_ext["attack_recall"] > m_base["attack_recall"] and m_ext["false_alarms"] <= m_base["false_alarms"]:
        print(f"\n-> The {n_new} LLM-authored signatures improved attack recall "
              f"without adding false alarms.\n   This is a measurable, "
              f"reproducible contribution: state it with these numbers.")
    else:
        print(f"\n-> The {n_new} LLM-authored signatures did not improve the "
              f"test result.\n   Report it honestly; the accept/reject audit "
              f"trail is still a finding.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    main(sys.argv[1])
