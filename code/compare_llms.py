"""
GridCopilot — Cross-model comparison of LLMs as knowledge engineers
====================================================================
    python compare_llms.py RelayMeasurementDataset.csv --dry-run
    python compare_llms.py RelayMeasurementDataset.csv

Answers: does the choice of proposing LLM change how many usable signatures
are authored, and how good they are?

WHY THIS IS CHEAP
-----------------
Comparing models on CLASSIFICATION would cost ~154 calls each and we already
know the answer is capped by the knowledge base (Section V-B). Comparing them
on AUTHORING costs ~3 calls each, because a model proposes six signatures per
call. Three models is ~9 calls total.

SPLIT DISCIPLINE — read this before running
-------------------------------------------
The sealed test split has already been opened once, for the result reported in
the paper. Evaluating several models on it and then reporting the best would be
selection on test, and would invalidate every number in the paper.

So this script NEVER touches the test split. It carves a VALIDATION split out
of the training data:

    full data ──┬── TEST (10,700)         sealed; untouched here
                └── TRAIN (42,796) ──┬── PROPOSE  (80% of train)
                                     └── VALIDATE (20% of train)

Every model proposes from PROPOSE statistics only, every proposal is scored on
PROPOSE (for admission) and then on VALIDATE (for the unbiased comparison the
table reports). The test split stays frozen at the already-published result.

If you later want a single model's extended KB evaluated on test, that number
already exists and must not be recomputed per model.
"""

import argparse
import json
import os
import sys
from typing import List, Dict

import numpy as np

from tcdr_descriptors import load_tcdr, compute_descriptors
from knowledge_base import Signature
from calibrate import (_signature_precision_recall, calibrate_signature,
                       kb_classify)
from llm_stage import (get_api_key, NO_KEY_MSG, GROQ_BASE_URL, call_llm,
                       load_frozen_kb)
from propose_signatures import (PROPOSAL_SYSTEM, build_proposal_prompt,
                                validate_syntax, validate_empirically,
                                sanitise_signature,
                                PER_PAIR_KEYS, SCALAR_KEYS)

LABEL_FAULT, LABEL_ATTACK = "fault", "attack"

DEFAULT_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z**2 / n
    c = p + z**2 / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((c - m) / d, (c + m) / d)


def evaluate_kb_on(descs, truth, sigs, weights, margin=0.05):
    preds = [kb_classify(d, sigs, margin, weights) for d in descs]
    n = len(truth)
    cov = [(p, t) for p, t in zip(preds, truth) if p is not None]
    tp = sum(1 for p, t in zip(preds, truth) if p == "attack" and t == "attack")
    fp = sum(1 for p, t in zip(preds, truth) if p == "attack" and t == "fault")
    n_atk = sum(1 for t in truth if t == "attack")
    return {
        "coverage": len(cov) / n if n else 0.0,
        "selective_acc": (sum(p == t for p, t in cov) / len(cov)) if cov else 0.0,
        "attack_recall": tp / max(n_atk, 1),
        "attack_recall_ci": wilson(tp, n_atk),
        "false_alarms": fp,
        "n_attack": n_atk,
    }


def run_model(model, client, kb, prop_desc, prop_lab, args):
    """Three rounds of proposals from one model. Returns (accepted, audit)."""
    import llm_stage
    system = PROPOSAL_SYSTEM.format(
        per_pair="\n".join(f"  {k}" for k in PER_PAIR_KEYS),
        scalars="\n".join(f"  {k}" for k in SCALAR_KEYS))

    covered = [kb_classify(d, list(kb.values()), 0.05, None) for d in prop_desc]
    gap = [(d, l) for d, l, c in zip(prop_desc, prop_lab, covered)
           if c is None or c != l]
    gap_d = [d for d, _ in gap]
    gap_l = [l for _, l in gap]

    accepted, audit, feedback = [], [], ""
    seen_ids = set(kb)
    # Deduplicate on CONDITIONS, not on id. A model that re-proposes the same
    # rule under a new name would otherwise be counted twice, overstating how
    # many distinct signatures it authored.
    seen_conditions = {frozenset(s.observable_conditions) for s in kb.values()}

    for rnd in range(1, args.rounds + 1):
        prompt = build_proposal_prompt(gap_d, gap_l, kb, args.ask, feedback)
        saved, llm_stage.SYSTEM_PROMPT = llm_stage.SYSTEM_PROMPT, system
        try:
            raw, _ = call_llm(client, model, prompt, json_mode=False,
                              sleep_s=args.sleep, max_tokens=args.max_tokens)
        except Exception as e:
            print(f"    round {rnd}: API failure ({type(e).__name__}) — "
                  f"recorded as a failed round, NOT as a model limitation")
            audit.append({"round": rnd, "api_error": str(e)[:200]})
            break
        finally:
            llm_stage.SYSTEM_PROMPT = saved

        i, j = (raw or "").find("["), (raw or "").rfind("]")
        if i < 0 or j < 0:
            print(f"    round {rnd}: no JSON array in response "
                  f"({len(raw or '')} chars)")
            audit.append({"round": rnd, "parse_failure": True,
                          "raw_len": len(raw or "")})
            continue
        try:
            proposals = json.loads(raw[i:j + 1])
        except Exception as e:
            print(f"    round {rnd}: malformed JSON ({e})")
            audit.append({"round": rnd, "json_error": str(e)[:200]})
            continue

        notes = []
        for p in proposals:
            sig = validate_syntax(p)
            if sig is None:
                audit.append({"round": rnd, "model": model,
                              "signature": str(p.get("signature_id", "?")),
                              "verdict": "REJECT malformed/unknown descriptor"})
                notes.append(f"{p.get('signature_id','?')}: malformed")
                continue
            if sig.signature_id in seen_ids:
                sig.signature_id += f"-R{rnd}"
            cond_key = frozenset(sig.observable_conditions)
            if cond_key in seen_conditions:
                audit.append({"round": rnd, "model": model,
                              "signature": sig.signature_id,
                              "conditions": sig.observable_conditions,
                              "verdict": "REJECT duplicate conditions",
                              "detail": "identical predicate set already present"})
                notes.append(f"{sig.signature_id}: REJECT duplicate of an "
                             f"existing signature")
                continue
            st, verdict = validate_empirically(
                sig, prop_desc, prop_lab,
                args.min_lift, args.min_fires, args.min_precision)
            detail = (f"fires={st.n_fired} prec={st.precision:.3f} "
                      f"lift={st.lift:.3f}" if st else "")
            audit.append({"round": rnd, "model": model,
                          "signature": sig.signature_id,
                          "conditions": sig.observable_conditions,
                          "verdict": verdict, "detail": detail,
                          "precision": st.precision if st else 0.0,
                          "lift": st.lift if st else 0.0,
                          "fires": st.n_fired if st else 0})
            notes.append(f"{sig.signature_id} "
                         f"[{' AND '.join(sig.observable_conditions)}] "
                         f"-> {verdict} {detail}")
            if verdict == "ACCEPT":
                tuned, _ = calibrate_signature(sig, prop_desc, prop_lab)
                st2, v2 = validate_empirically(
                    tuned, prop_desc, prop_lab,
                    args.min_lift, args.min_fires, args.min_precision)
                if v2 == "ACCEPT":
                    tuned_key = frozenset(tuned.observable_conditions)
                    if tuned_key in seen_conditions:
                        # calibration collapsed it onto an existing rule
                        notes.append(f"{tuned.signature_id}: REJECT — "
                                     f"calibration made it identical to an "
                                     f"existing signature")
                        continue
                    tuned = sanitise_signature(tuned)
                    accepted.append((tuned, st2))
                    seen_ids.add(tuned.signature_id)
                    seen_conditions.add(tuned_key)
        feedback = ("\n".join(notes) +
                    "\n\nPropose DIFFERENT signatures. Rejected ones were too "
                    "broad (low lift) or too narrow (too few fires).")
    return accepted, audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--ask", type=int, default=6)
    ap.add_argument("--min-lift", type=float, default=0.50)
    ap.add_argument("--min-fires", type=int, default=30)
    ap.add_argument("--min-precision", type=float, default=0.80)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--val-frac", type=float, default=0.20,
                    help="fraction of TRAIN held out for unbiased comparison")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the split sizes and cost estimate, no API calls")
    ap.add_argument("--out", default="llm_comparison.json")
    args = ap.parse_args()

    kb = load_frozen_kb()
    print(f"Seed knowledge base: {len(kb)} signatures")

    W, labels = load_tcdr(args.csv)
    test_idx = set(np.load("tcdr_test_indices.npy").tolist())
    train_idx = [i for i in range(len(W)) if i not in test_idx]

    # Split TRAIN into PROPOSE / VALIDATE, stratified, fixed seed.
    rng = np.random.default_rng(1234)
    tr_atk = [i for i in train_idx if labels[i] == LABEL_ATTACK]
    tr_flt = [i for i in train_idx if labels[i] == LABEL_FAULT]
    rng.shuffle(tr_atk); rng.shuffle(tr_flt)
    n_va_a = int(len(tr_atk) * args.val_frac)
    n_va_f = int(len(tr_flt) * args.val_frac)
    val_idx = tr_atk[:n_va_a] + tr_flt[:n_va_f]
    prop_idx = tr_atk[n_va_a:] + tr_flt[n_va_f:]

    print(f"TEST    {len(test_idx):>6}  SEALED — not touched by this script")
    print(f"PROPOSE {len(prop_idx):>6}  statistics shown to the LLMs, "
          f"admission scored here")
    print(f"VALIDATE{len(val_idx):>6}  unbiased comparison reported here")

    print("\nComputing descriptors...")
    prop_desc = [compute_descriptors(W[i]).flat() for i in prop_idx]
    prop_lab = [labels[i] for i in prop_idx]
    val_desc = [compute_descriptors(W[i]).flat() for i in val_idx]
    val_lab = [labels[i] for i in val_idx]

    base_sigs = list(kb.values())
    base = evaluate_kb_on(val_desc, val_lab, base_sigs, None)
    print(f"\nBaseline KB on VALIDATE: coverage {base['coverage']:.4f}  "
          f"attack recall {base['attack_recall']:.4f}  "
          f"false alarms {base['false_alarms']}")

    est_calls = len(args.models) * args.rounds
    print(f"\n{len(args.models)} models x {args.rounds} rounds = "
          f"~{est_calls} API calls total (~{est_calls * 3500:,} tokens)")
    if args.dry_run:
        print("\nDRY RUN — no API calls made.")
        return

    key = get_api_key(args.api_key)
    if not key:
        sys.exit(NO_KEY_MSG)
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("pip install openai")
    client = OpenAI(api_key=key, base_url=GROQ_BASE_URL)

    results, all_audit = {}, []
    for model in args.models:
        print(f"\n===== {model} =====")
        accepted, audit = run_model(model, client, kb, prop_desc, prop_lab, args)
        all_audit.extend(audit)
        n_prop = sum(1 for a in audit if "verdict" in a)
        print(f"  proposed {n_prop}, accepted {len(accepted)}")
        for sig, st in accepted:
            print(f"    {sig.signature_id}: "
                  f"{' AND '.join(sig.observable_conditions)}")
            print(f"      propose-split prec={st.precision:.3f} "
                  f"lift={st.lift:.3f} fires={st.n_fired}")

        # Unbiased evaluation on VALIDATE
        ext = base_sigs + [s for s, _ in accepted]
        weights = {}
        for s in base_sigs:
            stx = _signature_precision_recall(s, prop_desc, prop_lab)
            weights[s.signature_id] = stx.lift
        for s, st in accepted:
            weights[s.signature_id] = st.lift
        ev = evaluate_kb_on(val_desc, val_lab, ext, weights)
        # Rationale-predicate consistency. The validator checks predicates;
        # the RATIONALE is what an operator reads, and nothing verifies it.
        # Flag the common case where the name asserts a direction the
        # predicate contradicts.
        def rationale_conflict(sig):
            name = sig.signature_id.upper()
            flags = []
            for cond in sig.observable_conditions:
                up = ">=" in cond or "> " in cond
                low = "<=" in cond or "< " in cond
                key = cond.split()[0].split(".")[-1].upper()
                base = cond.split()[0].split(".")[0].upper().replace("_", "-")
                if up and ("LOW-" in name or "LOW_" in name) and base[:6] in name.replace("_", "-"):
                    flags.append(f"name says LOW but predicate is '{cond}'")
                if low and ("HIGH-" in name or "HIGH_" in name) and base[:6] in name.replace("_", "-"):
                    flags.append(f"name says HIGH but predicate is '{cond}'")
            return flags

        gen = []
        for s, st in accepted:
            stv = _signature_precision_recall(s, val_desc, val_lab)
            conflicts = rationale_conflict(s)
            if conflicts:
                print(f"    RATIONALE CONFLICT in {s.signature_id}:")
                for c in conflicts:
                    print(f"      {c}")
            gen.append({"rationale_conflicts": conflicts,
                        "signature": s.signature_id,
                        "propose_precision": st.precision,
                        "validate_precision": stv.precision,
                        "propose_lift": st.lift,
                        "validate_lift": stv.lift,
                        "validate_fires": stv.n_fired})
        api_failed = any("api_error" in a for a in audit)
        results[model] = {"n_proposed": n_prop, "n_accepted": len(accepted),
                          "validate": ev, "signatures": gen,
                          "api_failed": api_failed,
                          "ran": n_prop > 0}
        print(f"  VALIDATE: coverage {ev['coverage']:.4f} "
              f"({ev['coverage']-base['coverage']:+.4f})  "
              f"attack recall {ev['attack_recall']:.4f} "
              f"({ev['attack_recall']-base['attack_recall']:+.4f})  "
              f"false alarms {ev['false_alarms']}")

    # ---- Refuse to print a comparison the data cannot support ----
    ran = {m: r for m, r in results.items() if r["ran"]}
    failed = [m for m, r in results.items() if not r["ran"]]
    if not ran:
        print("\n" + "=" * 78)
        print("NO COMPARISON TABLE — not one model was successfully queried.")
        print("=" * 78)
        for m in failed:
            why = ("API authentication/transport failure"
                   if results[m]["api_failed"] else "no parsable proposals")
            print(f"  {m:<26} {why}")
        print("\nThe per-model rows would otherwise all show the SAME numbers,")
        print("because with zero accepted signatures each 'extended' KB is just")
        print("the baseline re-scored with lift weights. That is not a result")
        print("and must not be read as one.")
        joined = json.dumps(all_audit)
        if "Authentication" in joined or "invalid_api_key" in joined or "401" in joined:
            print("\nThe key in use is invalid or revoked. Pass a working one:")
            print("  python compare_llms.py <csv> --api-key gsk_...")
        json.dump({"baseline": base, "models": results, "audit": all_audit,
                   "comparison_valid": False},
                  open(args.out, "w"), indent=2, default=float)
        print(f"\nWrote {args.out} with comparison_valid=false.")
        sys.exit(1)
    if failed:
        print(f"\nWARNING: {len(failed)} model(s) did not run and are "
              f"EXCLUDED from the table below:")
        for m in failed:
            print(f"  {m}")

    # ---- Comparison table ----
    print("\n" + "=" * 78)
    print("CROSS-MODEL COMPARISON — all numbers on the VALIDATE split")
    print("=" * 78)
    print(f"{'model':<26}{'prop':>5}{'acc':>5}{'cover':>9}{'Δcover':>9}"
          f"{'atk rec':>9}{'Δrec':>8}{'FA':>4}")
    print("-" * 78)
    print(f"{'(baseline KB, 6 sigs)':<26}{'-':>5}{'-':>5}"
          f"{base['coverage']:>9.4f}{'-':>9}"
          f"{base['attack_recall']:>9.4f}{'-':>8}{base['false_alarms']:>4}")
    for m, r in ran.items():
        ev = r["validate"]
        print(f"{m:<26}{r['n_proposed']:>5}{r['n_accepted']:>5}"
              f"{ev['coverage']:>9.4f}{ev['coverage']-base['coverage']:>+9.4f}"
              f"{ev['attack_recall']:>9.4f}"
              f"{ev['attack_recall']-base['attack_recall']:>+8.4f}"
              f"{ev['false_alarms']:>4}")

    if any(r["n_accepted"] for r in ran.values()):
        print("\nPer-signature generalisation (propose -> validate precision):")
    else:
        print("\nNo signature was accepted by any model, so every row above is")
        print("the baseline KB re-scored. There is no model difference to read.")
    for m, r in ran.items():
        for g in r["signatures"]:
            drop = g["propose_precision"] - g["validate_precision"]
            flag = "  <-- overfit?" if drop > 0.15 else ""
            print(f"  {m:<24} {g['signature']:<28} "
                  f"{g['propose_precision']:.3f} -> "
                  f"{g['validate_precision']:.3f}  "
                  f"(fires {g['validate_fires']}){flag}")

    json.dump({"baseline": base, "models": results, "audit": all_audit,
               "comparison_valid": True,
               "models_excluded": failed},
              open(args.out, "w"), indent=2, default=float)
    print(f"\nWrote {args.out}")
    print("\nNOTE: these are VALIDATE-split numbers. The sealed test result in "
          "the paper\nstands as published and must NOT be recomputed per "
          "model — choosing a model\nby test performance would be selection on "
          "test.")


if __name__ == "__main__":
    main()