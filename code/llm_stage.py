"""
GridCopilot — Stages 2, 3 and 4 on real data
=============================================
    python llm_stage.py RelayMeasurementDataset.csv --list-models
    python llm_stage.py RelayMeasurementDataset.csv --dry-run
    python llm_stage.py RelayMeasurementDataset.csv --n 200 --sleep 1.0

Stage 2  retrieval of candidate signatures
Stage 3  grounded LLM inference (Groq, OpenAI-compatible endpoint)
Stage 4  deterministic verification of the LLM's cited evidence

WHERE THE LLM IS APPLIED
------------------------
The knowledge base already covers 93.2% of events at 1.000 attack precision.
Running the LLM on all of them would burn budget re-deriving answers the rules
already produce correctly. The LLM's job is the GAP: the ~7% where no signature
fires, plus the 129 attacks the rules miss.

    --mode abstained   events where the KB abstains  (the deployment case)
    --mode sample      stratified sample of all events (for ablations)

RETRIEVAL NOTE (affects the paper)
----------------------------------
Section III specifies dense embedding retrieval. This implements retrieval by
DETERMINISTIC NEAR-SATISFACTION instead: each signature is scored by how close
the event's descriptors come to satisfying its predicates. Reasons: the KB is
small (6-9 entries, so top-k over an embedding index is pointless), it needs no
second API, and it is reproducible. Section III must be updated to describe
what the code actually does -- do not leave the dense-retrieval claim in the
paper unless you implement it.

TEST-SET DISCIPLINE
-------------------
Uses the frozen calibrated_kb_tcdr.json and the saved test indices. The KB was
calibrated on train only. Do not edit signatures after reading these results.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np

from tcdr_descriptors import load_tcdr, compute_descriptors, textualize, PHASES
from knowledge_base import (Signature, LLMOutput, verify, evaluate_signature,
                            evaluate_predicate, PredicateResult)
from calibrate import parse_predicate, descriptor_values

# Groq (the inference provider) -- OpenAI-compatible endpoint.
# NOT xAI's Grok. Different company, different API.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Groq's lineup changes often and older defaults such as
# llama-3.3-70b-versatile are being deprecated on the Free/Developer tiers,
# so run --list-models to see what YOUR account can actually serve rather
# than trusting this default.
DEFAULT_MODEL = "openai/gpt-oss-120b"

# Approximate published rates for the default model; override with --price-in
# / --price-out. The free tier may cover this run entirely -- the estimate is
# an upper bound, not a bill.
PRICE_IN_PER_M = 0.15
PRICE_OUT_PER_M = 0.60



def get_api_key(cli_key: Optional[str] = None) -> Optional[str]:
    """Find the Groq key: --api-key, then GROQ_API_KEY, then a key file.

    Environment variables are surprisingly fragile on Windows -- setx only
    affects future shells, and some Python launchers start a fresh
    environment -- so a plain file alongside the script is the reliable
    fallback. Put the key in groq_key.txt (one line) and add that file to
    .gitignore; never commit it.
    """
    if cli_key:
        return cli_key.strip()
    env = os.environ.get("GROQ_API_KEY")
    if env and env.strip():
        return env.strip()
    for name in ("groq_key.txt", ".groq_key", ".env"):
        if os.path.exists(name):
            # utf-8-sig strips a BOM, which Notepad writes by default and
            # which otherwise corrupts the first character of the key.
            for line in open(name, encoding="utf-8-sig"):
                line = line.strip().strip('"').strip("'")
                if not line or line.startswith("#"):
                    continue
                if line.upper().startswith("GROQ_API_KEY"):
                    line = line.split("=", 1)[-1].strip().strip('"').strip("'")
                if line.startswith("gsk_"):
                    return line
    return None


NO_KEY_MSG = """No Groq API key found. Use any one of:

  1. A file next to this script (most reliable on Windows):
         echo gsk_your_key_here > groq_key.txt
  2. Pass it directly:
         python llm_stage.py data.csv --list-models --api-key gsk_your_key
  3. Environment variable (must be set in the SAME shell you run from):
         $env:GROQ_API_KEY = "gsk_your_key"
"""


# ---------------------------------------------------------------------------
# Load the frozen knowledge base
# ---------------------------------------------------------------------------

def load_frozen_kb(path="calibrated_kb_tcdr.json") -> Dict[str, Signature]:
    if not os.path.exists(path):
        sys.exit(f"{path} not found. Run run_tcdr.py first to calibrate and freeze.")
    with open(path) as f:
        rows = json.load(f)
    sigs = [Signature(**r) for r in rows]
    return {s.signature_id: s for s in sigs}


# ---------------------------------------------------------------------------
# Stage 2 — retrieval by near-satisfaction
# ---------------------------------------------------------------------------

def _predicate_closeness(pred: str, desc: Dict[str, float]) -> float:
    """1.0 if satisfied, decaying toward 0 as the descriptor falls short.

    Gives the LLM the signatures that ALMOST apply as well as those that do,
    so it can reason about near-misses -- while the verifier still admits only
    signatures whose predicates strictly hold.
    """
    quant, key, op, thr = parse_predicate(pred)
    res = evaluate_predicate(pred, desc)
    if res is PredicateResult.SATISFIED:
        return 1.0
    if res is PredicateResult.UNDEFINED:
        return 0.0
    vals = [v for v in descriptor_values(quant, key, desc) if not np.isnan(v)]
    if not vals:
        return 0.0
    scale = max(abs(thr), 1e-6)
    if op in (">=", ">"):
        gap = (thr - max(vals)) / scale
    elif op in ("<=", "<"):
        gap = (min(vals) - thr) / scale
    else:
        gap = min(abs(v - thr) for v in vals) / scale
    return float(max(0.0, 1.0 - min(abs(gap), 1.0)))


def retrieve(desc: Dict[str, float], kb: Dict[str, Signature], k: int = 5):
    scored = []
    for sid, sig in kb.items():
        s = np.mean([_predicate_closeness(c, desc) for c in sig.observable_conditions])
        scored.append((float(s), sid))
    scored.sort(reverse=True)
    return [sid for _, sid in scored[:k]]


# ---------------------------------------------------------------------------
# Stage 3 — prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a power system protection analyst assisting a human \
operator. You classify a transformer differential relay event as either a \
genuine FAULT or a cyber ATTACK (false data injection).

You reason ONLY from the numeric descriptors provided and the candidate \
signatures supplied as evidence. You may not invent signatures, and you may \
not appeal to knowledge outside the supplied evidence.

Rules, which are checked automatically after you answer:
1. Every signature_id you cite MUST appear in the CANDIDATE EVIDENCE list.
2. A signature only supports your label if the event's descriptors actually \
satisfy ALL of that signature's stated conditions. Check each condition \
against the numbers before citing it.
3. Your label MUST match the event_class of the signatures you cite.
4. If no candidate signature's conditions are genuinely satisfied, answer \
"insufficient_evidence". This is the correct answer when evidence is absent, \
and is strongly preferred over a guess.

Respond with ONLY a JSON object, no prose and no code fences:
{"label": "fault" | "attack" | "insufficient_evidence",
 "confidence": <float 0-1>,
 "cited_signature_ids": [<ids from the candidate list>],
 "justification": "<2-3 sentences citing specific descriptor values>"}"""


def build_prompt(window: np.ndarray, desc_obj, retrieved: List[str],
                 kb: Dict[str, Signature], desc_flat: Dict[str, float]) -> str:
    parts = [textualize(window, desc_obj, include_raw=False), "", "=== CANDIDATE EVIDENCE ==="]
    for sid in retrieved:
        sig = kb[sid]
        parts.append(f"\n{sid}  (event_class: {sig.event_class})")
        parts.append(f"  rationale: {sig.discriminative_rationale}")
        parts.append("  conditions that must ALL hold:")
        for c in sig.observable_conditions:
            parts.append(f"    - {c}")
    parts.append("\nClassify this event following the rules. JSON only.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Stage 3 — the API call
# ---------------------------------------------------------------------------

def call_llm(client, model: str, prompt: str, max_retries: int = 4,
             json_mode: bool = True, sleep_s: float = 0.0,
             max_tokens: int = 1200):
    """Call the chat endpoint. Retries with exponential backoff.

    json_mode uses Groq's JSON mode, which constrains the response to valid
    JSON. Note this constrains SYNTAX only -- it cannot make the model cite
    real signatures or satisfy predicates. That is Stage 4's job, and the gap
    between "valid JSON" and "verified evidence" is exactly the paper's point.
    """
    delay = 2.0
    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            r = client.chat.completions.create(**kwargs)
            if sleep_s:
                time.sleep(sleep_s)
            usage = getattr(r, "usage", None)
            return r.choices[0].message.content, (
                getattr(usage, "prompt_tokens", 0) if usage else 0,
                getattr(usage, "completion_tokens", 0) if usage else 0,
            )
        except Exception as e:
            msg = str(e)
            # Reasoning models (gpt-oss) emit thinking tokens before the
            # answer, which breaks Groq's JSON-mode validator and returns
            # json_validate_failed with an empty generation. Falling back to
            # plain prompting is correct: parse_llm_json already extracts the
            # outermost JSON object from surrounding prose.
            if json_mode and ("response_format" in msg or "json_object" in msg
                              or "json_validate_failed" in msg
                              or "Failed to validate JSON" in msg):
                print("    JSON mode rejected by this model; "
                      "falling back to plain prompting")
                json_mode = False
                continue
            if attempt == max_retries - 1:
                raise
            # 429 = rate limit; the free tier is limited, so back off harder
            wait = delay * (3 if "429" in msg or "rate" in msg.lower() else 1)
            print(f"    retry {attempt+1} after {type(e).__name__}; waiting {wait:.0f}s")
            time.sleep(wait)
            delay *= 2


def parse_llm_json(text: str) -> Optional[LLMOutput]:
    """Parse the model's JSON. A malformed response is a FORMAT FAILURE and is
    recorded as such -- never silently repaired, since repairing it would hide
    a real reliability property of the unconstrained model."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        t = t[4:] if t.lower().startswith("json") else t
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < 0:
        return None
    try:
        d = json.loads(t[i:j + 1])
        return LLMOutput(
            label=str(d.get("label", "")).strip().lower(),
            confidence=float(d.get("confidence", 0.0)),
            cited_signature_ids=[str(s) for s in d.get("cited_signature_ids", [])],
            justification=str(d.get("justification", "")),
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--n", type=int, default=200, help="events to process")
    ap.add_argument("--mode", choices=["abstained", "sample"], default="abstained")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--k", type=int, default=5, help="signatures retrieved")
    ap.add_argument("--dry-run", action="store_true", help="estimate cost, no API calls")
    ap.add_argument("--list-models", action="store_true",
                    help="ask your Groq account which models it can serve, then exit")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds between calls; use ~1.0 on the free tier")
    ap.add_argument("--no-json-mode", action="store_true",
                    help="disable JSON mode (required for gpt-oss reasoning models)")
    ap.add_argument("--max-tokens", type=int, default=1200,
                    help="completion budget; reasoning models need >=1000")
    ap.add_argument("--out", default="llm_results.jsonl")
    ap.add_argument("--api-key", default=None,
                    help="Groq key; overrides GROQ_API_KEY and groq_key.txt")
    args = ap.parse_args()

    if args.list_models:
        key = get_api_key(args.api_key)
        if not key:
            sys.exit(NO_KEY_MSG)
        # Use the OpenAI SDK rather than urllib: Cloudflare fingerprints
        # clients and rejects Python-urllib's default User-Agent outright
        # (that is the "error code: 1010" 403 -- a browser-signature ban,
        # nothing to do with the API key).
        print(f"Key loaded: {key[:7]}...{key[-4:]}  (length {len(key)})")
        if not key.startswith("gsk_"):
            print("WARNING: Groq keys start with 'gsk_'. This one does not.")
        try:
            from openai import OpenAI
        except ImportError:
            sys.exit("pip install openai")
        try:
            models = OpenAI(api_key=key, base_url=GROQ_BASE_URL).models.list()
        except Exception as e:
            msg = str(e)
            print(f"\nRequest failed: {type(e).__name__}")
            print(msg[:800])
            if "401" in msg or "invalid_api_key" in msg:
                print("\n-> Key invalid or revoked. Make a new one at "
                      "https://console.groq.com")
            elif "1010" in msg or "403" in msg:
                print("\n-> Still a Cloudflare block. Try a different network "
                      "(mobile hotspot/VPN);\n   if the console playground "
                      "works but this does not, it is network-level.")
            return
        data = {"data": [{"id": m.id} for m in models.data]}
        print("Models your account can serve:")
        for m in sorted(d.get("id", "?") for d in data.get("data", [])):
            print("  ", m)
        print("\nPick one with --model <id>. Prefer a text model with JSON "
              "mode; skip whisper/tts/guard models.")
        return

    kb = load_frozen_kb()
    print(f"Frozen KB: {len(kb)} signatures")

    W, labels = load_tcdr(args.csv)
    test_idx = np.load("tcdr_test_indices.npy")
    print(f"Test split: {len(test_idx)} events")

    print("Computing descriptors for the test split...")
    desc_objs = {i: compute_descriptors(W[i]) for i in test_idx}
    desc_flats = {i: d.flat() for i, d in desc_objs.items()}

    # Select which events the LLM handles
    if args.mode == "abstained":
        from calibrate import kb_classify
        with open("signature_weights.json") as f:
            weights = json.load(f)
        sigs = list(kb.values())
        targets = [i for i in test_idx
                   if kb_classify(desc_flats[i], sigs, 0.05, weights) is None]
        print(f"KB abstained on {len(targets)} test events -- the LLM's job")
    else:
        rng = np.random.default_rng(0)
        atk = [i for i in test_idx if labels[i] == "attack"]
        flt = [i for i in test_idx if labels[i] == "fault"]
        half = max(1, args.n // 2)
        targets = (list(rng.choice(atk, min(half, len(atk)), replace=False)) +
                   list(rng.choice(flt, min(half, len(flt)), replace=False)))
        # SHUFFLE. Without this the list is all attacks then all faults, so a
        # run that stops early (rate limit, Ctrl+C) yields a single-class
        # sample and every class-conditional metric is undefined.
        rng.shuffle(targets)
        print(f"Stratified sample: {len(targets)} events "
              f"(shuffled so partial runs stay balanced)")

    # Take a stratified slice, not the first n by index: the abstained set is
    # ~95% fault, so a plain head() yields almost no attacks and any
    # attack-side metric computed on it is noise.
    if len(targets) > args.n:
        rng2 = np.random.default_rng(0)
        t_atk = [i for i in targets if labels[i] == "attack"]
        t_flt = [i for i in targets if labels[i] == "fault"]
        keep_atk = t_atk[:args.n // 2]
        keep_flt = list(rng2.choice(t_flt, min(len(t_flt), args.n - len(keep_atk)),
                                    replace=False))
        targets = keep_atk + keep_flt
    targets = [int(i) for i in targets[:args.n]]
    n_atk = sum(1 for i in targets if labels[i] == "attack")
    print(f"Processing {len(targets)} events ({n_atk} attack, {len(targets)-n_atk} fault)")

    # Cost estimate from one real prompt
    i0 = targets[0]
    sample_prompt = build_prompt(W[i0], desc_objs[i0],
                                 retrieve(desc_flats[i0], kb, args.k), kb,
                                 desc_flats[i0])
    est_in = (len(SYSTEM_PROMPT) + len(sample_prompt)) / 4
    est_out = 120
    cost = (est_in * len(targets) / 1e6 * PRICE_IN_PER_M +
            est_out * len(targets) / 1e6 * PRICE_OUT_PER_M)
    print(f"\nEstimated: ~{est_in:.0f} input tokens/event, "
          f"~{est_out} output -> ~${cost:.2f} total for {len(targets)} events")

    if args.dry_run:
        print("\n--- DRY RUN: sample prompt ---")
        print(sample_prompt[:1800])
        print("\n(no API calls made)")
        return

    key = get_api_key(args.api_key)
    if not key:
        sys.exit(NO_KEY_MSG)
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("pip install openai")
    client = OpenAI(api_key=key, base_url=GROQ_BASE_URL)
    print(f"Using model: {args.model}  (--list-models to see alternatives)")

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["index"])
                except Exception:
                    pass
        print(f"Resuming: {len(done)} already done")

    tin = tout = 0
    consecutive_failures = 0
    with open(args.out, "a") as fout:
        for n, i in enumerate(targets, 1):
            if i in done:
                continue
            retrieved = retrieve(desc_flats[i], kb, args.k)
            prompt = build_prompt(W[i], desc_objs[i], retrieved, kb, desc_flats[i])
            try:
                raw, (a, b) = call_llm(client, args.model, prompt,
                                       json_mode=not args.no_json_mode,
                                       sleep_s=args.sleep,
                                       max_tokens=args.max_tokens)
            except Exception as e:
                print(f"  [{n}/{len(targets)}] FAILED: {e}")
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    sys.exit("\n5 consecutive failures -- aborting rather than "
                             "burning the whole run.\nTry: --no-json-mode, a "
                             "different --model, or a larger --max-tokens.")
                continue
            consecutive_failures = 0
            tin += a; tout += b
            out = parse_llm_json(raw)
            if out is None:
                rec = {"index": i, "truth": labels[i], "retrieved": retrieved,
                       "format_failure": True, "raw": raw[:500]}
            else:
                vr = verify(out, retrieved, desc_flats[i], kb)
                rec = {"index": i, "truth": labels[i], "retrieved": retrieved,
                       "format_failure": False,
                       "llm_label": out.label, "confidence": out.confidence,
                       "cited": out.cited_signature_ids,
                       "justification": out.justification,
                       "verified": vr.verified, "verified_label": vr.label,
                       "failure_reason": vr.failure_reason,
                       "citation_validity": vr.citation_validity,
                       "surviving": vr.surviving_signatures}
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            if n % 20 == 0 or n == len(targets):
                spent = tin/1e6*PRICE_IN_PER_M + tout/1e6*PRICE_OUT_PER_M
                print(f"  [{n}/{len(targets)}]  spent ~${spent:.2f}")

    print(f"\nWrote {args.out}.  Tokens: {tin} in / {tout} out  "
          f"(~${tin/1e6*PRICE_IN_PER_M + tout/1e6*PRICE_OUT_PER_M:.2f})")
    print("Now run:  python score_llm.py")


if __name__ == "__main__":
    main()