"""
GridCopilot — Score the LLM stage
==================================
    python score_llm.py [llm_results.jsonl]

Computes the Section IV-E metrics, including the explanation-faithfulness
family that prior LLM-for-power-systems work asserts but does not measure.

Reported:
  * format failure rate        -- malformed JSON, the unconstrained baseline's
                                  silent reliability cost
  * citation validity rate     -- cited IDs were actually retrieved
  * predicate satisfaction     -- cited signatures' conditions truly hold
  * label-evidence consistency -- label matches surviving evidence
  * VERIFIED EXPLANATION RATE  -- passes all three; the composite score
  * coverage / selective accuracy / full-coverage accuracy
  * the guardrail's effect: accuracy of RAW LLM labels vs VERIFIED labels

The last comparison is the paper's central claim. If verification does not
change accuracy or precision, the verifier is decorative and the contribution
collapses -- so it is printed prominently, not buried.
"""

import json
import sys
from collections import Counter


def main(path="llm_results.jsonl"):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    if not recs:
        sys.exit("No records found.")
    n = len(recs)
    print(f"=== GridCopilot LLM stage — {n} events ===")
    truth = Counter(r["truth"] for r in recs)
    print(f"ground truth: {dict(truth)}\n")

    fmt_fail = [r for r in recs if r.get("format_failure")]
    ok = [r for r in recs if not r.get("format_failure")]
    print(f"format failure rate:        {len(fmt_fail)/n:.4f}  ({len(fmt_fail)}/{n})")

    if not ok:
        sys.exit("All responses malformed — check the model name and prompt.")

    # An "insufficient_evidence" answer cites nothing BY DESIGN -- that is the
    # instructed behaviour when no signature applies, not a hallucination.
    # Counting it as a citation-validity failure conflates correct abstention
    # with fabrication and understates the model's faithfulness.
    claimed = [r for r in ok if r.get("llm_label") in ("fault", "attack")]
    cite_ok = [r for r in claimed if r.get("citation_validity")]
    verified = [r for r in ok if r.get("verified")]
    insuff = [r for r in ok if r.get("llm_label") == "insufficient_evidence"]

    # Predicate satisfaction: of those with valid citations and a real label,
    # how many had at least one cited signature whose conditions actually hold
    attempted = [r for r in ok if r.get("llm_label") in ("fault", "attack")]
    pred_ok = [r for r in attempted if r.get("surviving")]

    print(f"citation validity rate:     "
          f"{len(cite_ok)/max(len(claimed),1):.4f}  "
          f"({len(cite_ok)}/{len(claimed)})   [over claimed labels only]")
    print(f"predicate satisfaction:     "
          f"{len(pred_ok)/max(len(attempted),1):.4f}  "
          f"({len(pred_ok)}/{len(attempted)})")
    print(f"abstained (insufficient):   {len(insuff)/len(ok):.4f}  "
          f"({len(insuff)}/{len(ok)})")
    print(f"VERIFIED EXPLANATION RATE:  {len(verified)/n:.4f}  ({len(verified)}/{n})")

    # Guard against the single-class sample that makes every class metric
    # meaningless -- say so loudly rather than printing 0.0000 precision.
    classes = set(r["truth"] for r in recs)
    if len(classes) < 2:
        print(f"\n*** WARNING: all {n} events are '{classes.pop()}'. ***")
        print("*** Class-conditional metrics below are NOT interpretable.  ***")
        print("*** Re-run with the shuffled sampler for a balanced set.    ***")

    # ---- Coverage and accuracy ----
    print(f"\ncoverage (verified / all):  {len(verified)/n:.4f}")
    if verified:
        corr = sum(1 for r in verified if r["verified_label"] == r["truth"])
        print(f"selective accuracy:         {corr/len(verified):.4f}  "
              f"({corr}/{len(verified)})")
        for cls in ("attack", "fault"):
            tp = sum(1 for r in verified
                     if r["verified_label"] == cls and r["truth"] == cls)
            fp = sum(1 for r in verified
                     if r["verified_label"] == cls and r["truth"] != cls)
            fn = sum(1 for r in recs if r["truth"] == cls and
                     not (r.get("verified") and r.get("verified_label") == cls))
            p = tp/(tp+fp) if tp+fp else 0.0
            rc = tp/(tp+fn) if tp+fn else 0.0
            f1 = 2*p*rc/(p+rc) if p+rc else 0.0
            print(f"  {cls:<7} precision={p:.4f}  recall={rc:.4f}  F1={f1:.4f}  "
                  f"(tp={tp} fp={fp} fn={fn})")

    # ---- THE CENTRAL COMPARISON ----
    print("\n=== Effect of the guardrail (the paper's main claim) ===")
    raw = [r for r in ok if r.get("llm_label") in ("fault", "attack")]
    if raw:
        raw_corr = sum(1 for r in raw if r["llm_label"] == r["truth"])
        raw_fp = sum(1 for r in raw
                     if r["llm_label"] == "attack" and r["truth"] == "fault")
        print(f"RAW LLM (no verification):  n={len(raw)}  "
              f"accuracy={raw_corr/len(raw):.4f}  false attack alarms={raw_fp}")
    if verified:
        v_corr = sum(1 for r in verified if r["verified_label"] == r["truth"])
        v_fp = sum(1 for r in verified
                   if r["verified_label"] == "attack" and r["truth"] == "fault")
        print(f"VERIFIED only:              n={len(verified)}  "
              f"accuracy={v_corr/len(verified):.4f}  false attack alarms={v_fp}")
    rejected = [r for r in ok if not r.get("verified")
                and r.get("llm_label") in ("fault", "attack")]
    if rejected:
        wrong = sum(1 for r in rejected if r["llm_label"] != r["truth"])
        print(f"REJECTED by verifier:       n={len(rejected)}  "
              f"of which actually wrong={wrong} ({wrong/len(rejected):.1%})")
        print("  (high % wrong => the verifier is catching genuine errors,")
        print("   low % => it is mostly rejecting correct answers, costing coverage)")

    # ---- Why verification failed ----
    fails = Counter()
    for r in ok:
        if not r.get("verified") and r.get("failure_reason"):
            fr = r["failure_reason"]
            key = ("citation validity" if "Citation validity" in fr else
                   "predicate satisfaction" if "Predicate satisfaction" in fr else
                   "label-evidence inconsistency" if "inconsistency" in fr else
                   "conflicting evidence" if "Conflicting" in fr else
                   "no evidence cited" if "No evidence" in fr else "other")
            fails[key] += 1
    if fails:
        print("\nVerification failure breakdown:")
        for k, v in fails.most_common():
            print(f"  {k:<32} {v}")

    # ---- Sample outputs for the paper ----
    print("\n=== Example verified explanation (for the paper) ===")
    for r in verified[:1]:
        print(f"truth={r['truth']}  label={r['verified_label']}  "
              f"conf={r.get('confidence')}")
        print(f"cited: {r.get('surviving')}")
        print(f"justification: {r.get('justification')}")
    print("\n=== Example rejected output (shows the guardrail working) ===")
    for r in rejected[:1]:
        print(f"truth={r['truth']}  raw label={r['llm_label']}  "
              f"cited={r.get('cited')}")
        print(f"rejected because: {r.get('failure_reason')}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "llm_results.jsonl")