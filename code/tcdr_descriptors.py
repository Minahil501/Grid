"""
GridCopilot — TCDR Descriptors (rebuilt for the real dataset)
==============================================================
Replaces the original descriptors.py assumptions with what the
LLMSmartGridTCDR data actually contains. Empirically established:

  * Shape (53496, 193) = 192 features + label.
  * Layout is TIME-MAJOR: reshape(-1, 32, 6).transpose(0, 2, 1).
  * 32 samples span EXACTLY ONE AC cycle (dominant FFT bin = 1).
    Harmonic h therefore lands on bin h. DC-offset decay is NOT
    measurable within a single cycle and has been removed.
  * Differential pairs, recovered by per-sample residual minimisation:
        ch0 <-> ch3   via DIFFERENCE
        ch1 <-> ch5   via SUM        (opposite CT polarity)
        ch2 <-> ch4   via DIFFERENCE
    Pairs are labelled A/B/C for predicate compatibility; the mapping to
    physical phases is UNKNOWN and no claim about it is made.
  * Data is globally min-max scaled to [0, 1], sign information lost.
    Every descriptor here is therefore SCALE-INVARIANT (ratios only).
    This is what kept the Test 3/Test 4 results clean.
  * Zero-current is per-channel (0.459-0.542), so centring is per-channel
    per-window, never a shared constant.

Validated discriminators on real data (AUC / RF macro-F1):
  * cross-channel amplitude inconsistency  -- AUC 0.818 single scalar
  * normalised roughness (discontinuity)   -- attacks ~2x faults
  * multi-phase differential activation    -- faults 1 hot pair, attacks 2-3
  * 7 physics descriptors -> RF macro-F1 0.9942 (beats 192 raw features)
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import numpy as np

# Pair label -> (channel_i, channel_j, combination mode)
PAIR_MAP: Dict[str, Tuple[int, int, str]] = {
    "A": (0, 3, "DIFF"),
    "B": (1, 5, "SUM"),
    "C": (2, 4, "DIFF"),
}
PHASES = list(PAIR_MAP)          # ['A','B','C'] -- matches predicate quantifiers
SAMPLES_PER_CYCLE = 32
EPS = 1e-9


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_tcdr(path: str, label_col: str = "label"):
    """Load RelayMeasurementDataset.csv -> (windows, labels).

    windows: (n, 6, 32) float array, raw (uncentred) channel-major.
    labels:  list of 'fault' | 'attack'  (0 -> fault, 1 -> attack)
    """
    import pandas as pd
    df = pd.read_csv(path)
    y = df[label_col].to_numpy().astype(int)
    X = df.drop(columns=[label_col]).to_numpy(dtype=float)
    n_feat = X.shape[1]
    if n_feat != SAMPLES_PER_CYCLE * 6:
        raise ValueError(
            f"Expected {SAMPLES_PER_CYCLE*6} feature columns, found {n_feat}."
        )
    W = X.reshape(-1, SAMPLES_PER_CYCLE, 6).transpose(0, 2, 1)
    labels = ["fault" if v == 0 else "attack" for v in y]
    return W, labels


def center(W: np.ndarray) -> np.ndarray:
    """Remove each channel's own DC level, per window.

    Uses the median rather than the mean: with one cycle per window an
    asymmetric transient biases the mean, and the median is the more
    robust estimate of the zero-current level.
    """
    return W - np.median(W, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------

@dataclass
class EventDescriptors:
    # Differential protection, per pair
    diff_restraint_ratio: Dict[str, float]    # Id/Ir -- the operating quantity
    pair_amplitude_ratio: Dict[str, float]    # smaller/larger side amplitude

    # Harmonic content of the differential (bins 2 and 5 of a 1-cycle window)
    second_harmonic_ratio: Dict[str, float]
    fifth_harmonic_ratio: Dict[str, float]

    # Physics-violation indicators
    roughness: Dict[str, float]               # mean|di/dt| normalised by RMS
    discontinuity: Dict[str, float]           # max|d2i| normalised by RMS
    waveform_residual: Dict[str, float]       # residual vs single sinusoid

    # Cross-channel consistency (the strongest validated discriminator)
    amplitude_spread: float                   # std/mean of the 6 channel RMS
    hot_pair_count: float                     # pairs with Id/Ir above threshold
    max_roughness: float
    mean_roughness: float

    def to_dict(self) -> Dict:
        return asdict(self)

    def flat(self) -> Dict[str, float]:
        out = {}
        for key, val in asdict(self).items():
            if isinstance(val, dict):
                for sub, v in val.items():
                    out[f"{key}.{sub}"] = float(v)
            else:
                out[key] = float(val)
        return out


def _rms(a: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(a ** 2, axis=-1))


def _harmonic_ratio(sig: np.ndarray, order: int, ref: float) -> float:
    """Harmonic `order` relative to the fundamental of a 1-cycle window.

    Returns NaN when the differential is too small for the ratio to mean
    anything -- NaN propagates to the verifier as UNDEFINED, never as a
    silently-satisfied condition.
    """
    if ref < EPS:
        return float("nan")
    spec = np.abs(np.fft.rfft(sig))
    if len(spec) <= order or spec[1] < EPS:
        return float("nan")
    if _rms(sig) / ref < 0.02:      # differential is essentially noise
        return float("nan")
    return float(spec[order] / spec[1])


def _residual_vs_sinusoid(sig: np.ndarray) -> float:
    """Residual after fitting one cycle of a sinusoid, normalised by RMS.

    A LOW value during a claimed fault is suspicious: genuine transients
    carry harmonic and noise content, so excessive purity suggests
    synthesised data.
    """
    n = len(sig)
    r = _rms(sig)
    if r < EPS:
        return float("nan")
    t = np.arange(n)
    w = 2 * np.pi / n                       # exactly one cycle
    design = np.column_stack([np.ones(n), np.cos(w * t), np.sin(w * t)])
    coef, *_ = np.linalg.lstsq(design, sig, rcond=None)
    return float(_rms(sig - design @ coef) / r)


def compute_descriptors(window: np.ndarray,
                        hot_threshold: float = 1.0) -> EventDescriptors:
    """Compute descriptors for one (6, 32) window. Centring is applied here."""
    if window.shape != (6, SAMPLES_PER_CYCLE):
        raise ValueError(f"Expected (6, {SAMPLES_PER_CYCLE}), got {window.shape}")
    Wc = window - np.median(window, axis=-1, keepdims=True)
    ch_rms = _rms(Wc)

    idr, par, h2, h5, rough, disc, resid = {}, {}, {}, {}, {}, {}, {}

    for name, (i, j, mode) in PAIR_MAP.items():
        a, b = Wc[i], Wc[j]
        d = a + b if mode == "SUM" else a - b
        restraint = (ch_rms[i] + ch_rms[j]) / 2
        idr[name] = float(_rms(d) / restraint) if restraint > EPS else 0.0

        lo, hi = sorted([ch_rms[i], ch_rms[j]])
        par[name] = float(lo / hi) if hi > EPS else 0.0

        h2[name] = _harmonic_ratio(d, 2, restraint)
        h5[name] = _harmonic_ratio(d, 5, restraint)

        # Physics-violation measures taken on the input-side channel
        s = Wc[i]
        r = ch_rms[i]
        if r > EPS:
            rough[name] = float(np.mean(np.abs(np.diff(s))) / r)
            d2 = np.diff(s, n=2)
            disc[name] = float(np.max(np.abs(d2)) / r) if len(d2) else 0.0
        else:
            rough[name] = disc[name] = float("nan")
        resid[name] = _residual_vs_sinusoid(s)

    spread = float(ch_rms.std() / (ch_rms.mean() + EPS))
    hot = float(sum(1 for v in idr.values() if v > hot_threshold))
    rough_vals = [v for v in rough.values() if not np.isnan(v)]

    return EventDescriptors(
        diff_restraint_ratio=idr,
        pair_amplitude_ratio=par,
        second_harmonic_ratio=h2,
        fifth_harmonic_ratio=h5,
        roughness=rough,
        discontinuity=disc,
        waveform_residual=resid,
        amplitude_spread=spread,
        hot_pair_count=hot,
        max_roughness=float(max(rough_vals)) if rough_vals else float("nan"),
        mean_roughness=float(np.mean(rough_vals)) if rough_vals else float("nan"),
    )


def descriptors_for_all(W: np.ndarray, hot_threshold: float = 1.0) -> List[Dict[str, float]]:
    return [compute_descriptors(w, hot_threshold).flat() for w in W]


# ---------------------------------------------------------------------------
# Textualisation for LLM input
# ---------------------------------------------------------------------------

def textualize(window: np.ndarray, desc: EventDescriptors,
               include_raw: bool = True) -> str:
    parts: List[str] = []
    if include_raw:
        parts.append("=== RAW RELAY MEASUREMENTS (1 AC cycle, 32 samples) ===")
        for name, (i, j, mode) in PAIR_MAP.items():
            for side, ch in (("input", i), ("output", j)):
                arr = " ".join(f"{v:.4f}" for v in window[ch])
                parts.append(
                    f"Differential pair {name}, transformer {side} side "
                    f"(channel {ch}): [{arr}]"
                )

    parts.append("\n=== DERIVED PROTECTION DESCRIPTORS ===")
    fmt = lambda v: "undefined" if (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"
    for name in PHASES:
        parts.append(
            f"Pair {name}: Id/Ir = {fmt(desc.diff_restraint_ratio[name])}, "
            f"amplitude ratio = {fmt(desc.pair_amplitude_ratio[name])}, "
            f"2nd harmonic = {fmt(desc.second_harmonic_ratio[name])}, "
            f"5th harmonic = {fmt(desc.fifth_harmonic_ratio[name])}, "
            f"roughness = {fmt(desc.roughness[name])}, "
            f"discontinuity = {fmt(desc.discontinuity[name])}, "
            f"waveform residual = {fmt(desc.waveform_residual[name])}"
        )
    parts.append(
        f"System-wide: cross-channel amplitude spread = {fmt(desc.amplitude_spread)}, "
        f"pairs with elevated differential = {int(desc.hot_pair_count)}, "
        f"max roughness = {fmt(desc.max_roughness)}, "
        f"mean roughness = {fmt(desc.mean_roughness)}"
    )
    return "\n".join(parts)
