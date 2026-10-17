"""
GridCopilot — Like-for-like comparison on one event subset
===========================================================
    python compare_systems.py RelayMeasurementDataset.csv llm_qwen.jsonl

The KB baseline was measured on all 10,700 test events (7% attack); the LLM
was measured on a balanced sample (~49% attack). Those numbers are NOT
comparable -- class balance alone moves precision and recall substantially,
and a reviewer will say so. This recomputes every system on the SAME events
the LLM actually saw.

Systems compared:
  1. KB-only        deterministic signatures, lift-weighted vote
  2. LLM+verifier   GridCopilot's Stage 3+4 output
  3. COMBINED       KB where it fires, LLM where the KB abstains
                    (the deployment configuration)
  4. Random Forest  black-box reference, trained on the TRAIN split only

No API calls. Costs nothing, runs in ~2 minutes.
"""

import json
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score

from tcdr_descriptors import load_tcdr, descriptors_for_all, compute_descriptors
from llm_stage import load_frozen_kb
from calibrate import kb_classify


def wilson(k, n, z=1.96):
    """Wilson score interval -- the right CI for a proportion at small n.

    Reported because a 154-event sample cannot support bare point estimates;
    'recall 0.74' and 'recall 0.74 [0.63, 0.83]' say very different things to
    a reviewer.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z**2 / n
    c = p + z**2 / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((c - m) / d, (c + m) / d)


def report(name, preds, truth):
    """preds: list of 'fault' | 'attack' | None (abstain)."""
    n = len(truth)
    cov = [(p, t) for p, t in zip(preds, truth) if p is not None]
    print(f"\n--- {name} ---")
    print(f"coverage: {len(cov)/n:.4f}  ({len(cov)}/{n})")
    if not cov:
        return
    yp = [p for p, _ in cov]
    yt = [t for _, t in cov]
    acc = sum(p == t for p, t in cov) / len(cov)
    lo, hi = wilson(sum(p == t for p, t in cov), len(cov))
    print(f"selective accuracy: {acc:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"selective balanced acc: {balanced_accuracy_score(yt, yp):.4f}")
    print(classification_report(yt, yp, digits=4, zero_division=0))
    # attack recall over ALL events, abstentions counted as misses
    tp = sum(1 for p, t in zip(preds, truth) if p == "attack" and t == "attack")
    n_atk = sum(1 for t in truth if t == "attack")
    fp = sum(1 for p, t in zip(preds, truth) if p == "attack" and t == "fault")
    lo, hi = wilson(tp, n_atk)
    print(f"attack recall over ALL events: {tp}/{n_atk} = {tp/max(n_atk,1):.4f}"
          f"   95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"false attack alarms: {fp}")


def main(csv_path, results_path):
    recs = [json.loads(l) for l in open(results_path) if l.strip()]
    idx = [r["index"] for r in recs]
    truth = [r["truth"] for r in recs]
    print(f"Comparing on {len(idx)} events "
          f"({truth.count('attack')} attack, {truth.count('fault')} fault)")

    W, labels = load_tcdr(csv_path)
    kb = load_frozen_kb()
    sigs = list(kb.values())
    weights = json.load(open("signature_weights.json"))

    flats = {i: compute_descriptors(W[i]).flat() for i in idx}

    # 1. KB only
    kb_preds = [kb_classify(flats[i], sigs, 0.05, weights) for i in idx]
    report("KB-only (deterministic signatures)", kb_preds, truth)

    # 2. LLM + verifier
    llm_preds = [r.get("verified_label") if r.get("verified") else None
                 for r in recs]
    report("LLM + verifier (GridCopilot Stage 3+4)", llm_preds, truth)

    # 3. Combined: KB first, LLM on KB abstentions
    comb = [k if k is not None else l for k, l in zip(kb_preds, llm_preds)]
    report("COMBINED (KB, then LLM on abstentions)", comb, truth)

    # ---- Where do KB and LLM disagree or both fail? ----
    # The combined system only helps if the LLM covers events the KB cannot.
    # If both abstain on the same events, the pipeline adds nothing, and that
    # is a result to report rather than to engineer around.
    both_abstain = [(i, t) for i, k, l, t in zip(idx, kb_preds, llm_preds, truth)
                    if k is None and l is None]
    llm_rescues = [(i, t) for i, k, l, t in zip(idx, kb_preds, llm_preds, truth)
                   if k is None and l is not None]
    kb_only_cov = [(i, t) for i, k, l, t in zip(idx, kb_preds, llm_preds, truth)
                   if k is not None and l is None]
    disagree = [(i, k, l, t) for i, k, l, t in zip(idx, kb_preds, llm_preds, truth)
                if k is not None and l is not None and k != l]

    print("\n=== Complementarity analysis ===")
    print(f"KB abstained, LLM RESCUED:      {len(llm_rescues)}  "
          f"(attacks: {sum(1 for _, t in llm_rescues if t == 'attack')})")
    print(f"KB abstained, LLM also abstained: {len(both_abstain)}  "
          f"(attacks: {sum(1 for _, t in both_abstain if t == 'attack')})")
    print(f"KB covered, LLM abstained:      {len(kb_only_cov)}")
    print(f"Both covered but DISAGREE:      {len(disagree)}")
    for i, k, l, t in disagree[:5]:
        print(f"   event {i}: KB={k}  LLM={l}  truth={t}")
    if not llm_rescues:
        print("\n-> The LLM recovered NONE of the KB's abstentions. When no")
        print("   signature's conditions hold, the LLM has nothing admissible")
        print("   to cite either, so it abstains on the same events. The")
        print("   combined pipeline therefore equals the KB alone on this set.")
        print("   Report this; do not present COMBINED as an improvement.")

    # ---- Random Forest reference, same events ----
    print("\nTraining Random Forest on the train split (descriptor features)...")
    test_idx = set(np.load("tcdr_test_indices.npy").tolist())
    train_idx = [i for i in range(len(W)) if i not in test_idx]
    rng = np.random.default_rng(42)
    train_sub = rng.choice(train_idx, min(12000, len(train_idx)), replace=False)

    # Fixed column order across train and test -- a dict's iteration order is
    # stable in CPython but relying on it across two builds is fragile.
    cols = sorted(flats[idx[0]].keys())

    def matrix(indices):
        rows = []
        for i in indices:
            f = compute_descriptors(W[i]).flat()
            rows.append([f[c] for c in cols])
        return np.nan_to_num(np.array(rows, dtype=float),
                             nan=0.0, posinf=0.0, neginf=0.0)

    Xtr = matrix(train_sub)
    ytr = [labels[i] for i in train_sub]
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                n_jobs=-1, random_state=42).fit(Xtr, ytr)
    Xte = matrix(idx)
    report("Random Forest (black box, no explanation)",
           list(rf.predict(Xte)), truth)

    print("\nRF always predicts (coverage 1.000) and gives no justification.")
    print("The paper's comparison is not accuracy alone but accuracy WITH an")
    print("auditable evidence trail -- state that explicitly in Section V.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])