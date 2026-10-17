"""
GridCopilot — Knowledge Base for LLMSmartGridTCDR
==================================================
Signatures rewritten against what the real data supports.

REMOVED from the original placeholder KB, with reasons:
  * FLT-003-DECAYING-DC-OFFSET  -- DC decay is not measurable in a
    single-cycle window; the descriptor no longer exists.
  * ATK-003-MISSING-DC-TRANSIENT -- tested and REJECTED. Amplitude-
    normalised offset gave AUC 0.474 (no information). The apparent
    signal came from amplitude leaking through the unnormalised mean.
  * FLT-004-UNBALANCED-FAULT-SEQUENCE -- sequence components need a
    known A/B/C mapping, which this dataset does not provide.
  * ATK-005-IMPLAUSIBLE-PHASE-GEOMETRY -- same reason.

ADDED, each grounded in a measured result:
  * ATK-006 cross-channel amplitude inconsistency (AUC 0.818)
  * ATK-007 multi-pair differential activation
    (faults: 6369/8000 windows with exactly 1 hot pair, 0 with >=2;
     attacks: 3436/3732 windows with >=2)
  * FLT-005 single-pair differential activation (the fault counterpart)

Thresholds remain PLACEHOLDERS. Run calibrate.py on the training split
before any test inference, then freeze.

PROVENANCE: 'literature' entries encode protection-engineering knowledge and
carry no leakage risk. 'training-derived' entries must be built from the
training split ONLY.
"""

from knowledge_base import Signature

SIGNATURES = [

    # ---------------- Fault signatures ----------------
    Signature(
        signature_id="FLT-001-INTERNAL-DIFFERENTIAL",
        event_class="fault",
        attack_subtype=None,
        observable_conditions=[
            "any.diff_restraint_ratio >= 1.0",
            "any.second_harmonic_ratio < 0.15",
        ],
        discriminative_rationale=(
            "Internal transformer fault: differential current rises well above "
            "the restraint quantity on the affected pair while second-harmonic "
            "content stays low, separating a true internal fault from inrush."
        ),
        provenance="literature",
    ),
    Signature(
        signature_id="FLT-002-MAGNETIZING-INRUSH",
        event_class="fault",
        attack_subtype=None,
        observable_conditions=[
            "any.diff_restraint_ratio >= 1.0",
            "any.second_harmonic_ratio >= 0.15",
        ],
        discriminative_rationale=(
            "Magnetising inrush produces apparent differential current with "
            "strong second-harmonic content; a benign energisation condition "
            "rather than an internal fault."
        ),
        provenance="literature",
    ),
    Signature(
        signature_id="FLT-005-SINGLE-PAIR-ACTIVATION",
        event_class="fault",
        attack_subtype=None,
        observable_conditions=[
            "hot_pair_count == 1",
            "amplitude_spread <= 0.40",
        ],
        discriminative_rationale=(
            "A genuine single-phase fault raises the differential on exactly "
            "one pair while the remaining pairs keep balanced, proportionate "
            "amplitudes -- the localisation a real physical fault produces."
        ),
        provenance="training-derived",
    ),
    Signature(
        signature_id="FLT-006-PHYSICAL-TRANSIENT-CONTENT",
        event_class="fault",
        attack_subtype=None,
        observable_conditions=[
            "max_roughness <= 0.35",
            "any.diff_restraint_ratio >= 0.8",
        ],
        discriminative_rationale=(
            "Genuine fault currents remain sample-to-sample continuous because "
            "circuit inductance opposes instantaneous change; moderate "
            "roughness alongside a real differential indicates a physical "
            "transient rather than substituted samples."
        ),
        provenance="training-derived",
    ),

    # ---------------- Attack signatures ----------------
    Signature(
        signature_id="ATK-001-DERIVATIVE-DISCONTINUITY",
        event_class="attack",
        attack_subtype="FDI",
        observable_conditions=[
            "max_roughness >= 0.55",
        ],
        discriminative_rationale=(
            "Inductance prevents instantaneous current change, so elevated "
            "normalised sample-to-sample variation indicates splice points "
            "where injected values replaced measured ones. Measured: attack "
            "windows show roughly twice the roughness of fault windows."
        ),
        provenance="training-derived",
    ),
    Signature(
        signature_id="ATK-006-CROSS-CHANNEL-AMPLITUDE-INCONSISTENCY",
        event_class="attack",
        attack_subtype="FDI",
        observable_conditions=[
            "amplitude_spread >= 0.45",
        ],
        discriminative_rationale=(
            "A physical transformer event preserves proportionate amplitude "
            "relationships across all six measured channels. An adversary "
            "altering a subset of reported values breaks those relationships, "
            "producing amplitude dispersion no single physical event explains. "
            "Strongest single discriminator measured on this dataset."
        ),
        provenance="training-derived",
    ),
    Signature(
        signature_id="ATK-007-MULTI-PAIR-ACTIVATION",
        event_class="attack",
        attack_subtype="FDI",
        observable_conditions=[
            "hot_pair_count >= 2",
        ],
        discriminative_rationale=(
            "Simultaneous elevated differential on two or more pairs is "
            "inconsistent with the localised single-phase faults this relay "
            "observes, and indicates coordinated manipulation of multiple "
            "measurement channels."
        ),
        provenance="training-derived",
    ),
    Signature(
        signature_id="ATK-002-SYNTHETIC-WAVEFORM-PURITY",
        event_class="attack",
        attack_subtype="FDI",
        observable_conditions=[
            "any.waveform_residual <= 0.02",
            "any.diff_restraint_ratio >= 1.0",
        ],
        discriminative_rationale=(
            "A claimed fault whose waveform fits an ideal sinusoid almost "
            "perfectly is suspicious: real fault transients carry harmonic and "
            "noise content. Excessive purity suggests synthesised measurements."
        ),
        provenance="literature",
    ),
    Signature(
        signature_id="ATK-004-PAIR-AMPLITUDE-MISMATCH",
        event_class="attack",
        attack_subtype="FDI",
        observable_conditions=[
            "any.pair_amplitude_ratio <= 0.45",
            "any.diff_restraint_ratio >= 0.8",
        ],
        discriminative_rationale=(
            "Input and output currents of one differential pair retain "
            "comparable magnitudes through a genuine event. A large mismatch "
            "within a pair suggests one side's reported values were altered "
            "independently of the other."
        ),
        provenance="training-derived",
    ),
]
