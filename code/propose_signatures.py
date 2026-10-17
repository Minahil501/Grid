"""
GridCopilot — LLM as Knowledge Engineer (Stage 6)
==================================================
    python propose_signatures.py RelayMeasurementDataset.csv --rounds 3

THE GAP THIS FILLS
------------------
Measured: the guardrail-constrained LLM cannot exceed its knowledge base by
construction. It may only cite existing signatures, so where none applies it
must abstain -- which is why it rescued 0 of the KB's 11 abstentions and
disagreed with the KB 0 times out of 154.

The fix is to change what the LLM is FOR. Instead of using the knowledge base
to classify, it EXTENDS the knowledge base: it inspects events the current
signatures cannot handle and proposes new predicates. Every proposal is then
validated deterministically -- parsed, evaluated over the training split, and
kept only if it clears precision, lift and fire-count floors.

The LLM supplies hypotheses. Arithmetic supplies the truth. Neither can be
replaced by the other, and a proposal that survives is verifiable by anyone
who reruns the code.

TEST-SET DISCIPLINE (non-negotiable)
------------------------------------
Proposals are generated from and validated on the TRAINING SPLIT ONLY. The
test indices saved by run_tcdr.py are excluded throughout. A signature tuned
against test events would invalidate every number in the paper.
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Optional

import re
import numpy as np

from tcdr_descriptors import load_tcdr, compute_descriptors, PHASES
from dataclasses import replace
from knowledge_base import Signature, evaluate_signature
from calibrate import (_signature_precision_recall, calibrate_signature,
                       kb_classify, parse_predicate)
from llm_stage import (get_api_key, NO_KEY_MSG, GROQ_BASE_URL, call_llm,
                       load_frozen_kb)

LABEL_FAULT, LABEL_ATTACK = "fault", "attack"

# Descriptor keys a proposal may reference. Anything else is rejected at parse
# time -- the LLM cannot invent a measurement that does not exist.
PER_PAIR_KEYS = ["diff_restraint_ratio", "pair_amplitude_ratio",
                 "second_harmonic_ratio", "fifth_harmonic_ratio",
                 "roughness", "discontinuity", "waveform_residual"]
SCALAR_KEYS = ["amplitude_spread", "hot_pair_count",
               "max_roughness", "mean_roughness"]


# ---------------------------------------------------------------------------
# Describe the gap to the LLM
# ---------------------------------------------------------------------------

def summarise(descs: List[Dict[str, float]], labels: List[str]) -> str:
    """Per-class distribution summary of the uncovered events.

    Summary statistics rather than raw rows: 32 numbers per event would
    swamp the prompt, and the LLM reasons better over distributions than
    over individual samples anyway.
    """
    keys = SCALAR_KEYS + [f"{k}.{p}" for k in PER_PAIR_KEYS for p in PHASES]
    lines = [f"{'descriptor':<32} {'FAULT p25/p50/p75':>28} "
             f"{'ATTACK p25/p50/p75':>28}"]
    for k in keys:
        out = []
        for cls in (LABEL_FAULT, LABEL_ATTACK):
            vals = [d[k] for d, l in zip(descs, labels)
                    if l == cls and not np.isnan(d.get(k, np.nan))]
            if len(vals) < 3:
                out.append("        (too few)         ")
                continue
            q = np.percentile(vals, [25, 50, 75])
            out.append(f"{q[0]:>8.3f} {q[1]:>8.3f} {q[2]:>8.3f}")
        lines.append(f"{k:<32} {out[0]:>28} {out[1]:>28}")
    return "\n".join(lines)


PROPOSAL_SYSTEM = """You are a power system protection engineer extending a \
knowledge base of discriminative signatures for transformer differential \
relays. Each signature separates genuine FAULTS from cyber ATTACKS (false \
data injection).

You will see distribution statistics for events the CURRENT signatures cannot \
classify. Your job is to propose NEW signatures that separate the classes on \
exactly these hard events.

PREDICATE GRAMMAR -- proposals violating it are discarded automatically:
    <descriptor> <op> <number>
    any.<per_pair_descriptor> <op> <number>     (true if ANY of pairs A,B,C)
    all.<per_pair_descriptor> <op> <number>     (true if ALL of pairs A,B,C)
  operators: >= <= > < == !=

PER-PAIR DESCRIPTORS (use with any./all. or with .A/.B/.C):
{per_pair}

SCALAR DESCRIPTORS (use bare, no prefix):
{scalars}

WHAT THE DESCRIPTORS MEAN:
- diff_restraint_ratio: differential current over restraint current. High
  means the pair's input and output currents disagree -- a real differential.
- pair_amplitude_ratio: smaller/larger amplitude within one pair. Near 1.0
  means balanced; low means one side's reported values differ sharply.
- second/fifth_harmonic_ratio: harmonic content of the differential. High 2nd
  harmonic indicates magnetising inrush rather than an internal fault.
- roughness / discontinuity: normalised sample-to-sample variation. Physical
  current cannot jump instantaneously, so high values suggest injected data.
- waveform_residual: residual against an ideal sinusoid. VERY LOW during a
  claimed fault is suspicious -- real transients are not that clean.
- amplitude_spread: dispersion of amplitude across all six channels. A
  physical event keeps them proportionate; tampering with a subset does not.
- hot_pair_count: pairs with elevated differential. Genuine faults here are
  single-phase, so counts above 1 are anomalous.

RULES:
1. Propose signatures that are PRECISE, not broad. A signature firing on
   everything is worthless -- it will be rejected for insufficient lift.
2. Use 1-3 conditions each. ALL conditions must hold for the signature to fire.
3. Pick thresholds from the percentile tables so the signature actually fires.
4. Give each a physical rationale. A threshold with no physical story is
   curve-fitting and will not be accepted into the knowledge base.
5. Propose signatures for BOTH classes where the statistics support it.

Respond with ONLY a JSON array, no prose:
[{{"signature_id": "ATK-0XX-SHORT-NAME" or "FLT-0XX-SHORT-NAME",
   "event_class": "attack" | "fault",
   "attack_subtype": "FDI" or null,
   "observable_conditions": ["<predicate>", ...],
   "discriminative_rationale": "<the physics, 1-2 sentences>"}}]"""


def build_proposal_prompt(descs, labels, existing: Dict[str, Signature],
                          n_ask: int, feedback: str = "") -> str:
    n_f = labels.count(LABEL_FAULT)
    n_a = labels.count(LABEL_ATTACK)
    parts = [
        f"The current knowledge base has {len(existing)} signatures:",
    ]
    for s in existing.values():
        parts.append(f"  {s.signature_id} ({s.event_class}): "
                     f"{' AND '.join(s.observable_conditions)}")
    parts.append(
        f"\nThese {n_f + n_a} TRAINING events are NOT classified by any of the "
        f"above ({n_f} fault, {n_a} attack).\n"
        f"Distribution of descriptors on exactly these hard events:\n")
    parts.append(summarise(descs, labels))
    if feedback:
        parts.append("\nFEEDBACK ON YOUR PREVIOUS PROPOSALS:\n" + feedback)
    parts.append(
        f"\nPropose {n_ask} new signatures that separate these classes. "
        f"Look for descriptors where the fault and attack percentiles barely "
        f"overlap, and set thresholds between them. JSON array only.")
    return "\n".join(parts)




# ---------------------------------------------------------------------------
# Post-calibration name and rationale repair
# ---------------------------------------------------------------------------

_NUM_IN_NAME = re.compile(r"-?\d+(?:\.\d+)?")
_DIRECTION_WORDS = re.compile(
    r"\b(ABOVE|BELOW|OVER|UNDER|GREATER|LESS|LOWER|HIGHER|EXCEEDS?|AT-LEAST|"
    r"AT-MOST|GT|LT|GTE|LTE)\b", re.IGNORECASE)


def sanitise_signature(sig: Signature) -> Signature:
    """Remove stale numbers from the id and restate the rationale from the
    CALIBRATED predicates.

    Two failures this prevents, both observed in real runs:

    1. The LLM embeds its proposed threshold in the name
       (FLT-014-FIFTH-HARMONIC-B-ABOVE-0.3), calibration then retunes the
       threshold to 0.0383, and the name is left asserting a number the
       signature no longer uses.
    2. The LLM's name asserts a direction its predicate contradicts
       (FLT-010-LOW-2ND-HARMONIC-PAIR-B with condition
       second_harmonic_ratio.B >= 0.393).

    Both matter because the id and rationale are what reaches the operator in
    the audit trail. A predicate can be arithmetically sound while the prose
    beside it describes different physics; validating only the predicate
    leaves that gap open.
    """
    name = sig.signature_id
    # strip trailing numeric claims and comparison words from the id
    parts = [p for p in name.split("-") if p]
    cleaned, index_kept = [], False
    for pos, p in enumerate(parts):
        if _NUM_IN_NAME.fullmatch(p):
            # The FIRST numeral directly after the FLT/ATK prefix is the
            # sequence index and must be preserved to keep ids unique; any
            # later numeral is a threshold the model wrote in, which
            # calibration has since changed.
            if pos == 1 and not index_kept and "." not in p:
                cleaned.append(p)
                index_kept = True
            continue
        if _DIRECTION_WORDS.fullmatch(p):
            continue                      # direction word tied to that number
        cleaned.append(p)
    new_id = "-".join(cleaned) if len(cleaned) >= 2 else name

    # restate the conditions verbatim so the prose cannot drift from them
    cond_text = " AND ".join(sig.observable_conditions)
    rationale = sig.discriminative_rationale.strip()
    # drop any numeral the model wrote into the prose; it may be pre-calibration
    rationale = _NUM_IN_NAME.sub("[threshold]", rationale)
    new_rationale = (f"{rationale} Calibrated decision rule: {cond_text}.")

    return replace(sig, signature_id=new_id,
                   discriminative_rationale=new_rationale)


# ---------------------------------------------------------------------------
# Validation — the part the LLM cannot influence
# ---------------------------------------------------------------------------

def validate_syntax(raw: dict) -> Optional[Signature]:
    """Reject anything that is not a well-formed, evaluable signature."""
    try:
        sid = str(raw["signature_id"]).strip()
        cls = str(raw["event_class"]).strip().lower()
        conds = [str(c).strip() for c in raw["observable_conditions"]]
        if cls not in (LABEL_FAULT, LABEL_ATTACK) or not conds:
            return None
        for c in conds:
            quant, key, op, val = parse_predicate(c)   # raises if malformed
            base = key.split(".")[0]
            if quant and base not in PER_PAIR_KEYS:
                return None
            if not quant and base not in SCALAR_KEYS and base not in PER_PAIR_KEYS:
                return None
        return Signature(
            signature_id=sid,
            event_class=cls,
            attack_subtype=raw.get("attack_subtype"),
            observable_conditions=conds,
            discriminative_rationale=str(raw.get("discriminative_rationale", "")),
            provenance="llm-proposed-train-validated",
        )
    except Exception:
        return None


def validate_empirically(sig: Signature, descs, labels,
                         min_lift: float, min_fires: int, min_precision: float):
    """Evaluate on TRAINING data. Returns (stats, verdict_string)."""
    try:
        st = _signature_precision_recall(sig, descs, labels)
    except KeyError as e:
        return None, f"unknown descriptor {e}"
    if st.n_fired < min_fires:
        return st, f"REJECT: fired only {st.n_fired} times (< {min_fires})"
    if st.precision < min_precision:
        return st, f"REJECT: precision {st.precision:.3f} < {min_precision}"
    if st.lift < min_lift:
        return st, f"REJECT: lift {st.lift:.3f} < {min_lift} (no better than base rate)"
    return st, "ACCEPT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--ask", type=int, default=6, help="proposals per round")
    ap.add_argument("--min-lift", type=float, default=0.50)
    ap.add_argument("--min-fires", type=int, default=30)
    ap.add_argument("--min-precision", type=float, default=0.80)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--out", default="calibrated_kb_extended.json")
    args = ap.parse_args()

    kb = load_frozen_kb()
    existing_ids = set(kb)
    print(f"Current KB: {len(kb)} signatures")

    W, labels = load_tcdr(args.csv)
    test_idx = set(np.load("tcdr_test_indices.npy").tolist())
    train_idx = [i for i in range(len(W)) if i not in test_idx]
    print(f"Train: {len(train_idx)}   Test: {len(test_idx)} (SEALED)")

    print("Computing training descriptors...")
    tr_desc = [compute_descriptors(W[i]).flat() for i in train_idx]
    tr_lab = [labels[i] for i in train_idx]

    weights = json.load(open("signature_weights.json"))
    sigs = list(kb.values())
    covered = [kb_classify(d, sigs, 0.05, weights) for d in tr_desc]

    # The gap: abstentions plus events the KB gets wrong. Both are failures
    # the new signatures should address.
    gap = [(d, l) for d, l, c in zip(tr_desc, tr_lab, covered)
           if c is None or c != l]
    gap_d = [d for d, _ in gap]
    gap_l = [l for _, l in gap]
    print(f"Training events the KB fails on: {len(gap)} "
          f"({gap_l.count('attack')} attack, {gap_l.count('fault')} fault)")
    if len(gap) < 20:
        sys.exit("Gap too small to learn from.")

    key = get_api_key(args.api_key)
    if not key:
        sys.exit(NO_KEY_MSG)
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("pip install openai")
    client = OpenAI(api_key=key, base_url=GROQ_BASE_URL)

    system = PROPOSAL_SYSTEM.format(
        per_pair="\n".join(f"  {k}" for k in PER_PAIR_KEYS),
        scalars="\n".join(f"  {k}" for k in SCALAR_KEYS))

    accepted: List[Signature] = []
    feedback = ""
    audit = []
    # Track whether the LLM was ever actually reached. A transport failure is
    # NOT evidence about the LLM's ability and must never be reported as one.
    rounds_completed = 0
    api_errors = []

    for rnd in range(1, args.rounds + 1):
        print(f"\n===== ROUND {rnd} =====")
        prompt = build_proposal_prompt(gap_d, gap_l, kb, args.ask, feedback)
        import llm_stage
        saved, llm_stage.SYSTEM_PROMPT = llm_stage.SYSTEM_PROMPT, system
        try:
            raw, _ = call_llm(client, args.model, prompt, json_mode=False,
                              sleep_s=args.sleep, max_tokens=args.max_tokens)
        except Exception as e:
            print(f"API call failed: {e}")
            api_errors.append(str(e))
            break
        finally:
            llm_stage.SYSTEM_PROMPT = saved

        # Save every raw response: a silent parse failure is indistinguishable
        # from a model that refused, truncated, or wrapped its answer in prose,
        # and those need different fixes.
        with open(f"raw_proposal_round{rnd}.txt", "w", encoding="utf-8") as f:
            f.write(raw or "(empty response)")
        i, j = raw.find("["), raw.rfind("]")
        if i < 0 or j < 0:
            print(f"No JSON array found. Response was {len(raw or '')} chars; "
                  f"saved to raw_proposal_round{rnd}.txt")
            print("--- first 600 chars ---")
            print((raw or "(empty)")[:600])
            print("-----------------------")
            if raw and len(raw) > 100 and "{" in raw:
                print("(looks like objects without array brackets -- "
                      "attempting line-wise recovery)")
                try:
                    objs = []
                    depth, start = 0, None
                    for pos, ch in enumerate(raw):
                        if ch == "{":
                            if depth == 0:
                                start = pos
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0 and start is not None:
                                objs.append(json.loads(raw[start:pos + 1]))
                                start = None
                    if objs:
                        proposals = objs
                        print(f"recovered {len(objs)} proposal objects")
                    else:
                        continue
                except Exception:
                    continue
            else:
                continue
        else:
            try:
                proposals = json.loads(raw[i:j + 1])
            except Exception as e:
                print(f"Malformed JSON: {e}  (saved to "
                      f"raw_proposal_round{rnd}.txt)")
                continue
        rounds_completed += 1
        print(f"LLM proposed {len(proposals)} signatures")

        notes = []
        for p in proposals:
            sig = validate_syntax(p)
            if sig is None:
                msg = f"{p.get('signature_id','?')}: REJECT malformed/unknown descriptor"
                print("  " + msg); notes.append(msg); continue
            if sig.signature_id in existing_ids:
                sig.signature_id += f"-R{rnd}"
            st, verdict = validate_empirically(
                sig, tr_desc, tr_lab,
                args.min_lift, args.min_fires, args.min_precision)
            detail = (f"fires={st.n_fired} prec={st.precision:.3f} "
                      f"lift={st.lift:.3f} rec={st.recall:.3f}" if st else "")
            msg = f"{sig.signature_id}: {verdict}  {detail}"
            print("  " + msg)
            notes.append(f"{sig.signature_id} [{' AND '.join(sig.observable_conditions)}] -> {verdict} {detail}")
            audit.append({"round": rnd, "signature": sig.signature_id,
                          "conditions": sig.observable_conditions,
                          "verdict": verdict, "detail": detail})
            if verdict == "ACCEPT":
                # Tighten thresholds on TRAIN, exactly as hand-written ones are
                tuned, st2 = calibrate_signature(sig, tr_desc, tr_lab)
                st3, v2 = validate_empirically(
                    tuned, tr_desc, tr_lab,
                    args.min_lift, args.min_fires, args.min_precision)
                if v2 == "ACCEPT":
                    before = tuned.signature_id
                    tuned = sanitise_signature(tuned)
                    if tuned.signature_id != before:
                        print(f"    -> renamed {before} -> "
                              f"{tuned.signature_id} (stale threshold in name)")
                    accepted.append(tuned)
                    existing_ids.add(tuned.signature_id)
                    weights[tuned.signature_id] = st3.lift
                    print(f"    -> calibrated, kept "
                          f"(prec={st3.precision:.3f} lift={st3.lift:.3f})")

        feedback = ("\n".join(notes) +
                    "\n\nPropose DIFFERENT signatures. Rejected ones were too "
                    "broad (low lift) or too narrow (too few fires). Aim for "
                    "thresholds between the fault and attack percentiles.")

    # Distinguish "the LLM failed at the task" from "we never reached the LLM".
    if rounds_completed == 0:
        print("\n===== RUN ABORTED: the LLM was never successfully queried =====")
        for e in api_errors[:3]:
            print("  " + e[:200])
        print("\nNO CONCLUSION CAN BE DRAWN about whether the LLM can author "
              "signatures.\nThis is a connection failure, not a result. "
              "Nothing was written; fix the\nAPI key and re-run.")
        joined = " ".join(api_errors)
        if "invalid_api_key" in joined or "401" in joined:
            print("\nThe key in use is invalid. On Windows the environment "
                  "variable takes priority\nover groq_key.txt, so a stale setx "
                  "value silently overrides the file:\n"
                  '  [Environment]::SetEnvironmentVariable("GROQ_API_KEY", $null, "User")\n'
                  "  Remove-Item Env:GROQ_API_KEY -ErrorAction SilentlyContinue")
        sys.exit(1)

    print(f"\n===== RESULT: {len(accepted)} signatures accepted =====")
    for s in accepted:
        print(f"  {s.signature_id} ({s.event_class}): "
              f"{' AND '.join(s.observable_conditions)}")
        print(f"     {s.discriminative_rationale}")

    if not accepted:
        print(f"\nNo proposal survived validation ({len(audit)} proposals "
              f"evaluated over {rounds_completed} completed round(s)).")
        print("This IS a result -- but only because the LLM was actually "
              "queried and its\nproposals measured. See proposal_audit.json "
              "for the reject reasons.")

    payload = [{"signature_id": s.signature_id, "event_class": s.event_class,
                "attack_subtype": s.attack_subtype,
                "observable_conditions": s.observable_conditions,
                "discriminative_rationale": s.discriminative_rationale,
                "provenance": s.provenance}
               for s in list(kb.values()) + accepted]
    json.dump(payload, open(args.out, "w"), indent=2)
    json.dump(weights, open("signature_weights_extended.json", "w"), indent=2)
    json.dump(audit, open("proposal_audit.json", "w"), indent=2)
    print(f"\nWrote {args.out} ({len(payload)} signatures), "
          f"signature_weights_extended.json, proposal_audit.json")
    print("proposal_audit.json is your paper's accept/reject table.")
    print("\nNext:  python eval_extended.py " + args.csv)


if __name__ == "__main__":
    main()