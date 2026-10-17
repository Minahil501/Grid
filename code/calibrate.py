"""
GridCopilot — Threshold Calibration and Coverage Analysis
==========================================================
Every threshold in knowledge_base.py ships as a placeholder. This module
tunes them on the TRAINING SPLIT ONLY and characterises the resulting
coverage/accuracy tradeoff.

Two outputs feed directly into the paper:

  1. Calibrated signature thresholds (Section IV-B: KB construction).
  2. A coverage vs. selective-accuracy curve (Section V figure), showing
     what the guardrail costs in coverage and buys in precision.

It also produces a KB-ONLY RULE BASELINE -- classification by signature
firing alone, no LLM. Reviewers will ask "does the LLM add anything over
your rules?" and this is the number that answers it. If the rule baseline
matches full GridCopilot, the LLM is decorative and you need to know that
before submission, not after.

CRITICAL: never call calibrate() on test data. Fit on train, freeze, evaluate.

Driven by run_tcdr.py, which handles loading and splitting.
"""

from dataclasses import dataclass, replace
from typing import List, Dict, Tuple, Callable, Optional
import math
import json
import re
import numpy as np

from tcdr_descriptors import PHASES
from knowledge_base import (
    Signature, SIGNATURES, get_kb, evaluate_signature,
    evaluate_predicate, PredicateResult, _PRED_RE,
)

LABEL_FAULT, LABEL_ATTACK = "fault", "attack"


# ---------------------------------------------------------------------------
# Predicate manipulation
# ---------------------------------------------------------------------------

def parse_predicate(pred: str) -> Tuple[Optional[str], str, str, float]:
    m = _PRED_RE.match(pred)
    if not m:
        raise ValueError(f"Malformed predicate: {pred!r}")
    return (m.group("quant"), m.group("key"),
            m.group("op"), float(m.group("val")))


def rebuild_predicate(quant, key, op, val) -> str:
    return f"{quant or ''}{key} {op} {val:.6g}"


def descriptor_values(quant: Optional[str], key: str,
                      desc_flat: Dict[str, float]) -> List[float]:
    """All descriptor values a predicate reads (one per phase if quantified)."""
    if quant:
        return [desc_flat[f"{key}.{p}"] for p in PHASES]
    return [desc_flat[key]]


# ---------------------------------------------------------------------------
# Signature-level calibration
# ---------------------------------------------------------------------------

@dataclass
class SignatureStats:
    signature_id: str
    event_class: str
    threshold_set: List[str]
    fire_rate: float          # fraction of all training events where it fires
    precision: float          # of firings, fraction matching its event_class
    recall: float             # of its class, fraction it fires on
    n_fired: int
    lift: float = 0.0         # (precision - base_rate) / (1 - base_rate)
    base_rate: float = 0.0    # prevalence of this signature's class


def _signature_precision_recall(sig: Signature,
                                descs: List[Dict[str, float]],
                                labels: List[str]) -> SignatureStats:
    fired = np.array([evaluate_signature(sig, d)["satisfied"] for d in descs])
    y = np.array([l == sig.event_class for l in labels])
    n_fired = int(fired.sum())
    precision = float((fired & y).sum() / n_fired) if n_fired else 0.0
    recall = float((fired & y).sum() / y.sum()) if y.sum() else 0.0
    base_rate = float(y.mean())
    # Lift: how far the signature closes the gap from base rate to certainty.
    # 0.0 means it tells you nothing you did not already know from prevalence;
    # 1.0 means every firing is correct. This, not raw precision, is what
    # makes a signature usable as evidence under class imbalance.
    lift = ((precision - base_rate) / (1.0 - base_rate)
            if (1.0 - base_rate) > 1e-9 and n_fired else 0.0)
    return SignatureStats(
        signature_id=sig.signature_id,
        event_class=sig.event_class,
        threshold_set=list(sig.observable_conditions),
        fire_rate=float(fired.mean()),
        precision=precision,
        recall=recall,
        n_fired=n_fired,
        lift=lift,
        base_rate=base_rate,
    )


def calibrate_signature(sig: Signature,
                        descs: List[Dict[str, float]],
                        labels: List[str],
                        beta: float = 0.5,
                        n_candidates: int = 25,
                        min_precision: float = 0.85,
                        min_fire_count: int = 10,
                        lift_alpha: float = 0.5) -> Tuple[Signature, SignatureStats]:
    """Tune each threshold in a signature by coordinate descent on F-beta.

    beta < 1 weights PRECISION over recall deliberately. In this architecture
    a signature that fires on the wrong class is worse than one that fires
    rarely: false evidence corrupts a verified classification (the operator
    trusts it), whereas missing evidence only causes abstention (the operator
    is told to look). The asymmetry is the whole safety argument, so the
    objective must reflect it.
    """
    y = np.array([l == sig.event_class for l in labels])

    # CLASS-IMBALANCE GUARD.
    # With a 93/7 split, a fault signature that fires on EVERY window scores
    # 0.93 precision and passes any absolute floor below that -- calibration
    # then degenerates to "always fire", which is useless as evidence and
    # drowns out genuine minority-class signatures in the vote. The floor must
    # therefore sit above the class's own base rate: we require the signature
    # to close at least `lift_alpha` of the gap between base rate and 1.0.
    base_rate = float(y.mean())
    effective_min_precision = max(
        min_precision,
        base_rate + (1.0 - base_rate) * lift_alpha,
    )

    def fbeta(pred_fire: np.ndarray) -> float:
        tp = float((pred_fire & y).sum())
        fp = float((pred_fire & ~y).sum())
        fn = float((~pred_fire & y).sum())
        if tp == 0:
            return 0.0
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        if p < effective_min_precision or tp < min_fire_count:
            return 0.0
        b2 = beta ** 2
        return (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) > 0 else 0.0

    conditions = list(sig.observable_conditions)

    def fire_mask(conds: List[str]) -> np.ndarray:
        trial = replace(sig, observable_conditions=conds)
        return np.array([evaluate_signature(trial, d)["satisfied"] for d in descs])

    best_score = fbeta(fire_mask(conditions))

    for _ in range(2):  # two coordinate-descent passes
        for i, cond in enumerate(conditions):
            quant, key, op, _ = parse_predicate(cond)

            # Candidate thresholds drawn from the empirical distribution of
            # the descriptor this predicate reads.
            pool: List[float] = []
            for d in descs:
                pool.extend(v for v in descriptor_values(quant, key, d)
                            if not math.isnan(v))
            if len(pool) < min_fire_count:
                continue
            qs = np.linspace(2, 98, n_candidates)
            candidates = sorted(set(np.percentile(pool, qs).tolist()))

            for cand in candidates:
                trial_conds = list(conditions)
                trial_conds[i] = rebuild_predicate(quant, key, op, cand)
                score = fbeta(fire_mask(trial_conds))
                if score > best_score:
                    best_score = score
                    conditions = trial_conds

    tuned = replace(sig, observable_conditions=conditions)
    return tuned, _signature_precision_recall(tuned, descs, labels)


def calibrate_kb(signatures: List[Signature],
                 descs: List[Dict[str, float]],
                 labels: List[str],
                 **kw) -> Tuple[List[Signature], List[SignatureStats]]:
    tuned, stats = [], []
    for sig in signatures:
        t, s = calibrate_signature(sig, descs, labels, **kw)
        tuned.append(t)
        stats.append(s)
    return tuned, stats


# ---------------------------------------------------------------------------
# KB-only rule baseline + coverage analysis
# ---------------------------------------------------------------------------

def kb_classify(desc: Dict[str, float],
                signatures: List[Signature],
                min_margin: float = 1.0,
                weights: Optional[Dict[str, float]] = None) -> Optional[str]:
    """Classify by signature firing alone. Returns None = abstain.

    Votes are LIFT-WEIGHTED when `weights` is supplied (map signature_id ->
    lift, measured on the training split). Counting raw firings gives a
    signature that fires indiscriminately the same say as one that fires only
    on genuine cases -- under heavy imbalance the former swamps the latter.
    Weighting by lift makes an uninformative signature contribute ~0.

    Abstains when nothing fires, or when the margin between classes is too
    small -- the rule-based analogue of GridCopilot's conflicting-evidence
    abstention.
    """
    counts = {LABEL_FAULT: 0.0, LABEL_ATTACK: 0.0}
    for sig in signatures:
        if sig.event_class in counts and evaluate_signature(sig, desc)["satisfied"]:
            w = 1.0 if weights is None else float(weights.get(sig.signature_id, 0.0))
            counts[sig.event_class] += w
    if counts[LABEL_FAULT] <= 0 and counts[LABEL_ATTACK] <= 0:
        return None
    top = max(counts, key=counts.get)
    other = LABEL_ATTACK if top == LABEL_FAULT else LABEL_FAULT
    if counts[top] - counts[other] < min_margin:
        return None
    return top


@dataclass
class CoverageReport:
    margin: float
    coverage: float
    selective_accuracy: float
    full_coverage_accuracy: float
    n_covered: int
    n_total: int


def coverage_curve(descs: List[Dict[str, float]],
                   labels: List[str],
                   signatures: List[Signature],
                   margins=(0.5, 1.0, 1.5),
                   weights: Optional[Dict[str, float]] = None) -> List[CoverageReport]:
    """Sweep the abstention margin to trace coverage vs selective accuracy.

    This is your Section V figure: the guardrail buys precision on covered
    events at the cost of coverage, and both numbers must be reported.
    """
    reports = []
    n = len(descs)
    for m in margins:
        preds = [kb_classify(d, signatures, min_margin=m, weights=weights)
                 for d in descs]
        covered = [(p, t) for p, t in zip(preds, labels) if p is not None]
        n_cov = len(covered)
        sel_acc = (sum(p == t for p, t in covered) / n_cov) if n_cov else 0.0
        full_acc = sum(p == t for p, t in zip(preds, labels)) / n if n else 0.0
        reports.append(CoverageReport(
            margin=m,
            coverage=n_cov / n if n else 0.0,
            selective_accuracy=sel_acc,
            full_coverage_accuracy=full_acc,
            n_covered=n_cov,
            n_total=n,
        ))
    return reports


def print_stats_table(stats: List[SignatureStats], header: str):
    print(f"\n{header}")
    print(f"{'signature_id':<46} {'class':<7} {'fires':>6} {'prec':>7} "
          f"{'rec':>7} {'LIFT':>7}")
    print("-" * 84)
    for s in stats:
        flag = "  <-- uninformative" if s.lift < 0.3 else ""
        print(f"{s.signature_id:<46} {s.event_class:<7} {s.n_fired:>6} "
              f"{s.precision:>7.3f} {s.recall:>7.3f} {s.lift:>7.3f}{flag}")


def print_coverage(reports: List[CoverageReport], header: str):
    print(f"\n{header}")
    print(f"{'margin':>7} {'coverage':>10} {'sel.acc':>10} {'full.acc':>10} {'covered':>10}")
    print("-" * 52)
    for r in reports:
        print(f"{r.margin:>7.1f} {r.coverage:>10.3f} {r.selective_accuracy:>10.3f} "
              f"{r.full_coverage_accuracy:>10.3f} {r.n_covered:>6}/{r.n_total}")


def export_kb(signatures: List[Signature], path: str):
    payload = [{
        "signature_id": s.signature_id,
        "event_class": s.event_class,
        "attack_subtype": s.attack_subtype,
        "observable_conditions": s.observable_conditions,
        "discriminative_rationale": s.discriminative_rationale,
        "provenance": s.provenance,
    } for s in signatures]
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nCalibrated KB written to {path} — freeze this before test inference.")