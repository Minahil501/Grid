"""
GridCopilot — Stage 2 & 4: Attack-Signature Knowledge Base and Verifier
=======================================================================
The knowledge base holds atomic, discriminative signature entries whose
`observable_conditions` are COMPUTABLE PREDICATES over the flat descriptor
namespace produced by descriptors.py.

This computability is the whole point. A signature whose condition is prose
("the waveform looks unnatural") cannot be verified deterministically, and a
verifier that cannot check deterministically is just a second model that can
also be wrong. Every predicate here evaluates to True/False/UNDEFINED against
real numbers.

LEAKAGE WARNING: entries with provenance='training-derived' MUST be built
only from the training split. Freeze this KB before any test inference.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
from enum import Enum
import math
import operator
import re

EventClass = Literal["fault", "attack", "ambiguous"]


class PredicateResult(Enum):
    """Three-valued logic. UNDEFINED is distinct from FALSE.

    If a descriptor is NaN (e.g. harmonic ratio with no fundamental), the
    predicate is UNDEFINED -- we cannot assert the condition holds, and we
    must not silently treat that as a satisfied or violated condition.
    Treating UNDEFINED as FALSE would let a model cite evidence that was
    never actually evaluable.
    """
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNDEFINED = "undefined"


OPERATORS = {
    ">=": operator.ge, "<=": operator.le,
    ">": operator.gt, "<": operator.lt,
    "==": operator.eq, "!=": operator.ne,
}

# Predicate grammar: "<descriptor_key> <op> <number>"
# e.g. "diff_restraint_ratio.A >= 0.3"
# The 'any.' / 'all.' prefixes quantify over phases A, B, C:
# e.g. "any.diff_restraint_ratio >= 0.3"
_PRED_RE = re.compile(
    r"^\s*(?P<quant>any\.|all\.)?(?P<key>[A-Za-z_][A-Za-z0-9_.]*)"
    r"\s*(?P<op>>=|<=|==|!=|>|<)\s*(?P<val>-?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*$"
)

PHASES = ["A", "B", "C"]  # pair labels; see tcdr_descriptors.PAIR_MAP


@dataclass
class Signature:
    signature_id: str
    event_class: EventClass
    attack_subtype: Optional[str]
    observable_conditions: List[str]
    discriminative_rationale: str
    provenance: str  # 'literature' or 'training-derived'
    retrieval_text: str = ""  # what gets embedded for dense retrieval

    def __post_init__(self):
        if not self.retrieval_text:
            self.retrieval_text = (
                f"{self.event_class} {self.attack_subtype or ''} "
                f"{self.discriminative_rationale}"
            )


def evaluate_predicate(pred: str, desc_flat: Dict[str, float]) -> PredicateResult:
    """Evaluate one predicate string against the flat descriptor namespace."""
    m = _PRED_RE.match(pred)
    if not m:
        raise ValueError(
            f"Malformed predicate: {pred!r}. "
            "Grammar: '[any.|all.]<descriptor_key> <op> <number>'"
        )
    quant, key, op, val = (m.group("quant"), m.group("key"),
                           m.group("op"), float(m.group("val")))
    fn = OPERATORS[op]

    if quant:
        keys = [f"{key}.{p}" for p in PHASES]
        results = []
        for k in keys:
            if k not in desc_flat:
                raise KeyError(f"Unknown descriptor '{k}' in predicate {pred!r}")
            v = desc_flat[k]
            results.append(PredicateResult.UNDEFINED if math.isnan(v)
                           else (PredicateResult.SATISFIED if fn(v, val)
                                 else PredicateResult.VIOLATED))
        if quant == "any.":
            if any(r is PredicateResult.SATISFIED for r in results):
                return PredicateResult.SATISFIED
            if all(r is PredicateResult.UNDEFINED for r in results):
                return PredicateResult.UNDEFINED
            return PredicateResult.VIOLATED
        else:  # all.
            if any(r is PredicateResult.UNDEFINED for r in results):
                return PredicateResult.UNDEFINED
            return (PredicateResult.SATISFIED
                    if all(r is PredicateResult.SATISFIED for r in results)
                    else PredicateResult.VIOLATED)

    if key not in desc_flat:
        raise KeyError(f"Unknown descriptor '{key}' in predicate {pred!r}")
    v = desc_flat[key]
    if math.isnan(v):
        return PredicateResult.UNDEFINED
    return (PredicateResult.SATISFIED if fn(v, val)
            else PredicateResult.VIOLATED)


def evaluate_signature(sig: Signature,
                       desc_flat: Dict[str, float]) -> Dict:
    """Evaluate all conditions of a signature. Satisfied only if ALL hold."""
    per_condition = {}
    for cond in sig.observable_conditions:
        per_condition[cond] = evaluate_predicate(cond, desc_flat)
    satisfied = all(r is PredicateResult.SATISFIED
                    for r in per_condition.values())
    return {
        "signature_id": sig.signature_id,
        "satisfied": satisfied,
        "conditions": {k: v.value for k, v in per_condition.items()},
    }


# ---------------------------------------------------------------------------
# EXAMPLE SIGNATURE ENTRIES
# ---------------------------------------------------------------------------
# These are STARTER entries demonstrating the schema and the reasoning style.
# The thresholds are placeholders drawn from standard protection practice and
# MUST be re-tuned on your training split before use -- see calibrate() below.
# You will need substantially more entries for real coverage.

SIGNATURES: List[Signature] = [

    # ---- Genuine fault signatures (literature-derived) ----
    Signature(
        signature_id="FLT-001-INTERNAL-DIFFERENTIAL",
        event_class="fault",
        attack_subtype=None,
        observable_conditions=[
            "any.diff_restraint_ratio >= 0.3",
            "any.second_harmonic_ratio < 0.15",
        ],
        discriminative_rationale=(
            "Internal transformer fault: differential current rises well above "
            "the restraint characteristic while second-harmonic content stays "
            "low, distinguishing a true internal fault from magnetizing inrush."
        ),
        provenance="literature",
    ),
    Signature(
        signature_id="FLT-002-MAGNETIZING-INRUSH",
        event_class="fault",
        attack_subtype=None,
        observable_conditions=[
            "any.diff_restraint_ratio >= 0.3",
            "any.second_harmonic_ratio >= 0.15",
        ],
        discriminative_rationale=(
            "Magnetizing inrush on energization produces apparent differential "
            "current with strong second-harmonic content; this is a benign "
            "condition rather than an internal fault, and relays restrain on it."
        ),
        provenance="literature",
    ),
    Signature(
        signature_id="FLT-003-DECAYING-DC-OFFSET",
        event_class="fault",
        attack_subtype=None,
        observable_conditions=[
            "any.dc_offset_decay > 0.005",
            "any.diff_restraint_ratio >= 0.2",
        ],
        discriminative_rationale=(
            "A genuine fault current contains a DC offset that decays with the "
            "circuit X/R time constant. Its presence is positive evidence of a "
            "real electromagnetic transient rather than injected data."
        ),
        provenance="literature",
    ),
    Signature(
        signature_id="FLT-004-UNBALANCED-FAULT-SEQUENCE",
        event_class="fault",
        attack_subtype=None,
        observable_conditions=[
            "negative_sequence_ratio >= 0.15",
            "any.diff_restraint_ratio >= 0.2",
        ],
        discriminative_rationale=(
            "Unbalanced faults (single-line-to-ground, line-to-line) produce "
            "negative-sequence current consistent with the affected phases; "
            "the sequence pattern corroborates a genuine asymmetric fault."
        ),
        provenance="literature",
    ),

    # ---- Attack signatures (physics-violation based) ----
    Signature(
        signature_id="ATK-001-DERIVATIVE-DISCONTINUITY",
        event_class="attack",
        attack_subtype="FDI",
        observable_conditions=[
            "any.derivative_discontinuity >= 0.8",
        ],
        discriminative_rationale=(
            "Circuit inductance prevents instantaneous change in current. A "
            "large normalized second difference indicates a splice point where "
            "injected samples were substituted for measured ones -- a physical "
            "impossibility in genuine relay data."
        ),
        provenance="literature",
    ),
    Signature(
        signature_id="ATK-002-SYNTHETIC-WAVEFORM-PURITY",
        event_class="attack",
        attack_subtype="FDI",
        observable_conditions=[
            "any.waveform_smoothness <= 0.01",
            "any.diff_restraint_ratio >= 0.3",
        ],
        discriminative_rationale=(
            "A fault condition with near-zero residual against an ideal "
            "sinusoid is suspicious: real fault transients carry harmonic and "
            "noise content. Excessive waveform purity during a claimed fault "
            "suggests synthetically generated measurements."
        ),
        provenance="literature",
    ),
    Signature(
        signature_id="ATK-003-MISSING-DC-TRANSIENT",
        event_class="attack",
        attack_subtype="FDI",
        observable_conditions=[
            "any.diff_restraint_ratio >= 0.5",
            "all.dc_offset_decay <= 0.001",
        ],
        discriminative_rationale=(
            "A large differential current with no decaying DC component in any "
            "phase is inconsistent with a real fault inception transient, and "
            "suggests the differential was fabricated by altering reported "
            "magnitudes rather than produced by a physical event."
        ),
        provenance="literature",
    ),
    Signature(
        signature_id="ATK-004-INCONSISTENT-IN-OUT-CORRELATION",
        event_class="attack",
        attack_subtype="FDI",
        observable_conditions=[
            "any.in_out_correlation >= -0.3",
            "any.in_out_correlation <= 0.3",
            "any.diff_restraint_ratio >= 0.3",
        ],
        discriminative_rationale=(
            "Input and output side currents of a transformer remain strongly "
            "correlated (positively or negatively per CT convention) even "
            "during faults. Near-zero correlation suggests one side's data was "
            "tampered with independently of the other."
        ),
        provenance="literature",
    ),
    Signature(
        signature_id="ATK-005-IMPLAUSIBLE-PHASE-GEOMETRY",
        event_class="attack",
        attack_subtype="FDI",
        observable_conditions=[
            "phase_balance_deviation >= 30.0",
            "zero_sequence_ratio <= 0.05",
        ],
        discriminative_rationale=(
            "Large departure from 120-degree phase separation without "
            "corresponding zero-sequence current is electrically inconsistent: "
            "genuine phase-angle disturbances of this magnitude produce "
            "sequence-component signatures that are absent here."
        ),
        provenance="literature",
    ),
]


def get_kb() -> Dict[str, Signature]:
    return {s.signature_id: s for s in SIGNATURES}


# ---------------------------------------------------------------------------
# Stage 4: The Verifier
# ---------------------------------------------------------------------------

@dataclass
class LLMOutput:
    """Structured output the LLM is constrained to produce."""
    label: str                 # 'fault' | 'attack' | 'insufficient_evidence'
    confidence: float
    cited_signature_ids: List[str]
    justification: str


@dataclass
class VerificationResult:
    verified: bool
    label: Optional[str]
    failure_reason: Optional[str]
    citation_validity: bool
    surviving_signatures: List[str] = field(default_factory=list)
    rejected_signatures: Dict[str, str] = field(default_factory=dict)
    condition_detail: Dict[str, Dict] = field(default_factory=dict)


def verify(llm_out: LLMOutput,
           retrieved_ids: List[str],
           desc_flat: Dict[str, float],
           kb: Dict[str, Signature]) -> VerificationResult:
    """Run the three-check verification of Section III-D.

    Check 1: citation validity  -- cited IDs were actually retrieved
    Check 2: predicate satisfaction -- cited signatures' conditions truly hold
    Check 3: label-evidence consistency -- label matches surviving evidence
    """
    # --- Check 1: citation validity ---
    retrieved = set(retrieved_ids)
    invalid = [sid for sid in llm_out.cited_signature_ids
               if sid not in retrieved or sid not in kb]
    if invalid:
        return VerificationResult(
            verified=False, label=None,
            failure_reason=(
                f"Citation validity failed: cited signature(s) {invalid} were "
                "not retrieved for this event or do not exist in the knowledge "
                "base. This is a hallucinated citation."
            ),
            citation_validity=False,
        )

    if not llm_out.cited_signature_ids:
        return VerificationResult(
            verified=False, label=None,
            failure_reason="No evidence cited; classification is ungrounded.",
            citation_validity=False,
        )

    # --- Check 2: predicate satisfaction ---
    surviving, rejected, detail = [], {}, {}
    for sid in llm_out.cited_signature_ids:
        ev = evaluate_signature(kb[sid], desc_flat)
        detail[sid] = ev["conditions"]
        if ev["satisfied"]:
            surviving.append(sid)
        else:
            unmet = [c for c, r in ev["conditions"].items() if r != "satisfied"]
            rejected[sid] = f"unmet conditions: {unmet}"

    if not surviving:
        return VerificationResult(
            verified=False, label=None,
            failure_reason=(
                "Predicate satisfaction failed: none of the cited signatures "
                "have conditions actually satisfied by this event's measured "
                "descriptors. The model cited inapplicable evidence."
            ),
            citation_validity=True,
            rejected_signatures=rejected,
            condition_detail=detail,
        )

    # --- Check 3: label-evidence consistency ---
    evidence_classes = {kb[sid].event_class for sid in surviving}
    if llm_out.label not in evidence_classes:
        return VerificationResult(
            verified=False, label=None,
            failure_reason=(
                f"Label-evidence inconsistency: model emitted '{llm_out.label}' "
                f"but surviving evidence supports {sorted(evidence_classes)}."
            ),
            citation_validity=True,
            surviving_signatures=surviving,
            rejected_signatures=rejected,
            condition_detail=detail,
        )

    if len(evidence_classes) > 1:
        return VerificationResult(
            verified=False, label=None,
            failure_reason=(
                "Conflicting evidence: surviving signatures support more than "
                f"one event class {sorted(evidence_classes)}. Escalating for "
                "manual review rather than arbitrating automatically."
            ),
            citation_validity=True,
            surviving_signatures=surviving,
            rejected_signatures=rejected,
            condition_detail=detail,
        )

    return VerificationResult(
        verified=True,
        label=llm_out.label,
        failure_reason=None,
        citation_validity=True,
        surviving_signatures=surviving,
        rejected_signatures=rejected,
        condition_detail=detail,
    )


def format_operator_report(result: VerificationResult,
                           llm_out: LLMOutput,
                           kb: Dict[str, Signature]) -> str:
    """Stage 5: the auditable operator-facing report."""
    lines = []
    if result.verified:
        lines.append(f"CLASSIFICATION: {result.label.upper()}  [VERIFIED]")
        lines.append(f"Model confidence: {llm_out.confidence:.2f}")
        lines.append("\nSupporting evidence (all conditions independently confirmed):")
        for sid in result.surviving_signatures:
            sig = kb[sid]
            lines.append(f"  [{sid}] {sig.discriminative_rationale}")
            for cond, res in result.condition_detail[sid].items():
                lines.append(f"      - {cond}  ->  {res}")
    else:
        lines.append("CLASSIFICATION: UNVERIFIED — MANUAL REVIEW REQUIRED")
        lines.append(f"Reason: {result.failure_reason}")
        if result.rejected_signatures:
            lines.append("\nRejected evidence:")
            for sid, why in result.rejected_signatures.items():
                lines.append(f"  [{sid}] {why}")
    return "\n".join(lines)