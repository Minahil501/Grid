"""
GridCopilot — Run calibration and KB-only evaluation on the real TCDR data.

    python run_tcdr.py RelayMeasurementDataset.csv

Produces, all on real data:
  * per-signature precision/recall before and after calibration
  * coverage vs selective-accuracy curve (Section V figure)
  * KB-only rule baseline on the held-out TEST split
  * calibrated_kb_tcdr.json  -- freeze this before LLM inference

Discipline enforced here:
  * stratified split (the data is 93% fault / 7% attack)
  * calibration sees the TRAINING split only
  * attack-class recall and macro-F1 reported, never bare accuracy
"""

import sys
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             classification_report)

from tcdr_descriptors import load_tcdr, descriptors_for_all
from signatures_tcdr import SIGNATURES
from calibrate import (calibrate_kb, coverage_curve, kb_classify,
                       _signature_precision_recall, print_stats_table,
                       print_coverage)


def evaluate_kb(descs, labels, signatures, margin=1.0, tag="", weights=None):
    preds = [kb_classify(d, signatures, min_margin=margin, weights=weights)
             for d in descs]
    covered = [(p, t) for p, t in zip(preds, labels) if p is not None]
    n = len(descs)
    cov = len(covered) / n if n else 0.0
    print(f"\n=== KB-only rule baseline {tag} (margin={margin}) ===")
    print(f"coverage: {cov:.3f}  ({len(covered)}/{n})")
    if covered:
        yt = [t for _, t in covered]
        yp = [p for p, _ in covered]
        print(f"selective balanced acc: {balanced_accuracy_score(yt, yp):.4f}")
        print(f"selective macro-F1:     {f1_score(yt, yp, average='macro'):.4f}")
        print(classification_report(yt, yp, digits=4, zero_division=0))
    # abstentions counted as errors -- the conservative bound
    full = [p if p is not None else "abstain" for p in preds]
    print(f"full-coverage accuracy (abstain=error): "
          f"{np.mean([p == t for p, t in zip(full, labels)]):.4f}")


def main(path):
    print("Loading", path)
    W, labels = load_tcdr(path)
    print(f"  {len(W)} windows   "
          f"fault={labels.count('fault')}  attack={labels.count('attack')}")

    print("\nComputing descriptors (a few minutes for 53k windows)...")
    descs = descriptors_for_all(W)

    idx = np.arange(len(descs))
    tr, te = train_test_split(idx, test_size=0.2, stratify=labels,
                              random_state=42)
    d_tr = [descs[i] for i in tr]; y_tr = [labels[i] for i in tr]
    d_te = [descs[i] for i in te]; y_te = [labels[i] for i in te]
    print(f"  train={len(tr)}  test={len(te)}  (stratified)")

    before = [_signature_precision_recall(s, d_tr, y_tr) for s in SIGNATURES]
    print_stats_table(before, "BEFORE calibration — TRAIN split")

    print("\nCalibrating on TRAIN only...")
    tuned, after = calibrate_kb(SIGNATURES, d_tr, y_tr)
    print_stats_table(after, "AFTER calibration — TRAIN split")

    # Drop signatures by LIFT, not raw precision. Under 93/7 imbalance a
    # fault signature firing on everything reaches 0.93 precision while
    # carrying no information; lift < 0.3 means it barely improves on simply
    # guessing the prevalent class, so it is not admissible evidence.
    MIN_LIFT, MIN_FIRES = 0.30, 20
    keep, weights = [], {}
    for s, st in zip(tuned, after):
        if st.n_fired >= MIN_FIRES and st.lift >= MIN_LIFT:
            keep.append(s)
            weights[s.signature_id] = st.lift
    dropped = [st.signature_id for s, st in zip(tuned, after) if s not in keep]
    if dropped:
        print(f"\nDropped (fired <{MIN_FIRES} times or lift <{MIN_LIFT}):")
        for sid in dropped:
            print("   -", sid)
    print(f"Retained {len(keep)} / {len(tuned)} signatures.")
    if not keep:
        print("\nNo signature survived. Widen the signatures before continuing.")
        return

    MARGINS = (0.05, 0.2, 0.4, 0.6)
    print_coverage(coverage_curve(d_tr, y_tr, keep, MARGINS, weights),
                   "Coverage curve — TRAIN (lift-weighted)")
    print_coverage(coverage_curve(d_te, y_te, keep, MARGINS, weights),
                   "Coverage curve — TEST (lift-weighted)")

    for m in (0.05, 0.4):
        evaluate_kb(d_te, y_te, keep, margin=m, tag="TEST", weights=weights)

    # Attack-evidence recall on its own: what the attack signatures catch
    # before any fault vote can outweigh them. The gap between this and the
    # voted recall above shows how much the vote rule is costing.
    atk = [s for s in keep if s.event_class == "attack"]
    from knowledge_base import evaluate_signature
    fired_any = [any(evaluate_signature(s, d)["satisfied"] for s in atk)
                 for d in d_te]
    n_atk = sum(1 for t in y_te if t == "attack")
    tp = sum(1 for f, t in zip(fired_any, y_te) if f and t == "attack")
    fp = sum(1 for f, t in zip(fired_any, y_te) if f and t == "fault")
    print(f"\n=== Attack signatures, union (no voting) ===")
    print(f"union recall on attacks: {tp}/{n_atk} = {tp/max(n_atk,1):.4f}")
    print(f"false positives on faults: {fp}  "
          f"(precision {tp/max(tp+fp,1):.4f})")

    payload = [{
        "signature_id": s.signature_id,
        "event_class": s.event_class,
        "attack_subtype": s.attack_subtype,
        "observable_conditions": s.observable_conditions,
        "discriminative_rationale": s.discriminative_rationale,
        "provenance": s.provenance,
    } for s in keep]
    with open("calibrated_kb_tcdr.json", "w") as f:
        json.dump(payload, f, indent=2)
    np.save("tcdr_test_indices.npy", te)
    with open("signature_weights.json", "w") as f:
        json.dump(weights, f, indent=2)
    print("\nWrote calibrated_kb_tcdr.json and tcdr_test_indices.npy")
    print("FREEZE the KB now — do not recalibrate after seeing test results.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])