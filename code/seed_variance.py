"""
GridCopilot — Seed variance of LLM signature proposal
======================================================
    python seed_variance.py RelayMeasurementDataset.csv --repeats 3 \
        --model openai/gpt-oss-20b --api-key gsk_...

Closes the "single run" limitation. Section V-D currently reports one run per
model, and we already know proposal is non-deterministic even at temperature 0,
so the acceptance rates in Table V are single draws. This repeats the whole
Stage 6 loop N times for one model and reports the distribution.

WHAT IT MEASURES
----------------
Per repeat: how many proposals were made, how many passed validation, and what
the resulting knowledge base achieves on the VALIDATE split. Across repeats:
mean +/- SD of each, plus how often the SAME predicate is rediscovered, which
is the more interesting quantity -- a rule found in every repeat is structure
in the data, not an artefact of one sampling path.

COST
----
repeats x rounds API calls. Three repeats of three rounds is nine calls,
roughly 30k tokens; well inside a daily free-tier budget for one model.

SPLIT DISCIPLINE
----------------
Identical to compare_llms.py: proposals are generated and admitted on the
PROPOSE split, reported on VALIDATE, and the sealed test split is never
touched.
"""

import argparse
import json
import statistics
import sys
from collections import Counter

import numpy as np

from tcdr_descriptors import load_tcdr, compute_descriptors
from calibrate import _signature_precision_recall, kb_classify
from llm_stage import get_api_key, NO_KEY_MSG, GROQ_BASE_URL, load_frozen_kb
from compare_llms import run_model, evaluate_kb_on

LABEL_FAULT, LABEL_ATTACK = "fault", "attack"


class Args:
    """run_model expects an argparse-like object."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--ask", type=int, default=6)
    ap.add_argument("--min-lift", type=float, default=0.50)
    ap.add_argument("--min-fires", type=int, default=30)
    ap.add_argument("--min-precision", type=float, default=0.80)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.20)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--out", default="seed_variance.json")
    a = ap.parse_args()

    kb = load_frozen_kb()
    W, labels = load_tcdr(a.csv)
    test_idx = set(np.load("tcdr_test_indices.npy").tolist())
    train_idx = [i for i in range(len(W)) if i not in test_idx]

    rng = np.random.default_rng(1234)          # SAME split as compare_llms.py
    tr_atk = [i for i in train_idx if labels[i] == LABEL_ATTACK]
    tr_flt = [i for i in train_idx if labels[i] == LABEL_FAULT]
    rng.shuffle(tr_atk); rng.shuffle(tr_flt)
    n_va_a, n_va_f = int(len(tr_atk) * a.val_frac), int(len(tr_flt) * a.val_frac)
    val_idx = tr_atk[:n_va_a] + tr_flt[:n_va_f]
    prop_idx = tr_atk[n_va_a:] + tr_flt[n_va_f:]

    print(f"TEST {len(test_idx)} SEALED | PROPOSE {len(prop_idx)} | "
          f"VALIDATE {len(val_idx)}")
    print("Computing descriptors...")
    prop_desc = [compute_descriptors(W[i]).flat() for i in prop_idx]
    prop_lab = [labels[i] for i in prop_idx]
    val_desc = [compute_descriptors(W[i]).flat() for i in val_idx]
    val_lab = [labels[i] for i in val_idx]

    base_sigs = list(kb.values())
    base = evaluate_kb_on(val_desc, val_lab, base_sigs, None)
    print(f"\nBaseline on VALIDATE: coverage {base['coverage']:.4f}  "
          f"attack recall {base['attack_recall']:.4f}\n")

    key = get_api_key(a.api_key)
    if not key:
        sys.exit(NO_KEY_MSG)
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("pip install openai")
    client = OpenAI(api_key=key, base_url=GROQ_BASE_URL)

    runargs = Args(rounds=a.rounds, ask=a.ask, min_lift=a.min_lift,
                   min_fires=a.min_fires, min_precision=a.min_precision,
                   max_tokens=a.max_tokens, sleep=a.sleep)

    repeats, predicate_counter = [], Counter()
    for r in range(1, a.repeats + 1):
        print(f"===== REPEAT {r}/{a.repeats}  ({a.model}) =====")
        accepted, audit = run_model(a.model, client, kb, prop_desc,
                                    prop_lab, runargs)
        n_prop = sum(1 for x in audit if "verdict" in x)
        if n_prop == 0:
            print("  no proposals parsed; repeat recorded as failed, "
                  "NOT as zero acceptance")
            repeats.append({"repeat": r, "failed": True})
            continue

        ext = base_sigs + [s for s, _ in accepted]
        weights = {}
        for s in base_sigs:
            weights[s.signature_id] = _signature_precision_recall(
                s, prop_desc, prop_lab).lift
        for s, st in accepted:
            weights[s.signature_id] = st.lift
        ev = evaluate_kb_on(val_desc, val_lab, ext, weights)

        for s, _ in accepted:
            predicate_counter[" AND ".join(sorted(s.observable_conditions))] += 1

        print(f"  proposed {n_prop}, accepted {len(accepted)} "
              f"({len(accepted)/n_prop:.1%})")
        print(f"  VALIDATE coverage {ev['coverage']:.4f}  "
              f"attack recall {ev['attack_recall']:.4f}  "
              f"FA {ev['false_alarms']}\n")
        repeats.append({"repeat": r, "failed": False, "n_proposed": n_prop,
                        "n_accepted": len(accepted),
                        "accept_rate": len(accepted) / n_prop,
                        "validate": ev,
                        "signatures": [s.signature_id for s, _ in accepted]})

    ok = [r for r in repeats if not r.get("failed")]
    if len(ok) < 2:
        print("Fewer than two successful repeats; no variance to report.")
        json.dump({"baseline": base, "repeats": repeats}, open(a.out, "w"),
                  indent=2, default=float)
        sys.exit(1)

    def ms(vals):
        return statistics.fmean(vals), (statistics.stdev(vals)
                                        if len(vals) > 1 else 0.0)

    print("=" * 66)
    print(f"SEED VARIANCE over {len(ok)} successful repeats — {a.model}")
    print("=" * 66)
    for label, vals in [
        ("proposals made", [r["n_proposed"] for r in ok]),
        ("proposals accepted", [r["n_accepted"] for r in ok]),
        ("acceptance rate", [r["accept_rate"] for r in ok]),
        ("VALIDATE coverage", [r["validate"]["coverage"] for r in ok]),
        ("VALIDATE attack recall", [r["validate"]["attack_recall"] for r in ok]),
        ("false attack alarms", [r["validate"]["false_alarms"] for r in ok]),
    ]:
        m, s = ms(vals)
        print(f"  {label:<26} {m:>8.4f}  +/- {s:.4f}   "
              f"(min {min(vals):.4f}, max {max(vals):.4f})")

    print("\nPredicate rediscovery across repeats:")
    print("  A rule found in every repeat reflects structure in the data;")
    print("  one found once may be an artefact of a single sampling path.")
    for pred, c in predicate_counter.most_common():
        mark = "  <-- found in ALL repeats" if c == len(ok) else ""
        print(f"  {c}/{len(ok)}  {pred[:72]}{mark}")

    json.dump({"baseline": base, "model": a.model, "repeats": repeats,
               "predicate_counts": dict(predicate_counter)},
              open(a.out, "w"), indent=2, default=float)
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
