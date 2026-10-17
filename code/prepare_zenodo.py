"""
GridCopilot — Assemble the Zenodo archive
==========================================
    python prepare_zenodo.py

Collects the code and result artifacts into gridcopilot_artifact/ and zips it,
while refusing to include two categories of file:

  * SECRETS -- groq_key.txt, .env, anything matching an API-key pattern. A
    Zenodo deposit is PERMANENT and public; a leaked key cannot be withdrawn.
  * THE DATASET -- LLMSmartGridTCDR is released by its authors for academic
    use and is not ours to redistribute. The archive points to it instead.

Every file is scanned for key-shaped strings before packaging, and the script
refuses to produce the zip if any are found.
"""

import hashlib
import os
import re
import shutil
import sys
import zipfile

OUT = "gridcopilot_artifact"

CODE = [
    "tcdr_descriptors.py", "knowledge_base.py", "signatures_tcdr.py",
    "calibrate.py", "run_tcdr.py", "llm_stage.py", "score_llm.py",
    "compare_systems.py", "propose_signatures.py", "eval_extended.py",
    "compare_llms.py", "seed_variance.py", "adaptive_adversary.py",
    "measure_latency.py", "edge_footprint.py", "make_figures.py",
    "prepare_zenodo.py",
]

RESULTS = [
    "calibrated_kb_tcdr.json", "calibrated_kb_extended.json",
    "signature_weights.json", "signature_weights_extended.json",
    "proposal_audit.json", "tcdr_test_indices.npy",
    "llm_qwen.jsonl", "llm_comparison.json", "seed_variance.json",
    "latency_results.json", "adversary_results.json", "edge_footprint.json",
]

GLOBS = ["raw_proposal_round*.txt"]

# Never package these, whatever else happens.
FORBIDDEN = {"groq_key.txt", ".groq_key", ".env", "RelayMeasurementDataset.csv"}

SECRET_PATTERNS = [
    (re.compile(r"gsk_[A-Za-z0-9]{20,}"), "Groq API key"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "OpenAI-style API key"),
    (re.compile(r"xai-[A-Za-z0-9]{20,}"), "xAI API key"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key"),
    (re.compile(r"ghp_[A-Za-z0-9]{30,}"), "GitHub token"),
]


def scan(path):
    """Return a list of (pattern_name, masked_match) found in a file."""
    try:
        text = open(path, "r", encoding="utf-8", errors="ignore").read()
    except Exception:
        return []
    hits = []
    for pat, name in SECRET_PATTERNS:
        for m in pat.findall(text):
            hits.append((name, m[:7] + "..." + m[-4:]))
    return hits


README = """# GridCopilot — code and result artifacts

Companion archive for the paper *GridCopilot: An LLM Knowledge Engineer with
Deterministic Validation for Explainable Cyberattack-versus-Fault
Discrimination in Power Systems*.

## What this archive is

Everything needed to reproduce every number in the paper, except the dataset,
which is not ours to redistribute (see below).

## Dataset (not included)

Experiments use **LLMSmartGridTCDR**, released by the authors of:

> A. M. Saber, S. Jafari, Z. Ouyang, P. Budnarain, A. Youssef and D. Kundur,
> "Large language models for detecting cyberattacks on smart grid protective
> relays," *IEEE Open Access Journal of Power and Energy*, vol. 13,
> pp. 135-144, 2026.

It is available from that work's authors for academic and non-commercial
research use, conditional on citing the paper above. This archive does not
redistribute it. Place `RelayMeasurementDataset.csv` in the working directory
before running anything.

## Reproducing the results

Run in this order. Steps marked (API) require a Groq API key in `groq_key.txt`;
all others are offline.

| Step | Command | Produces |
|---|---|---|
| 1 | `python run_tcdr.py RelayMeasurementDataset.csv` | calibrated knowledge base, frozen test split |
| 2 | `python propose_signatures.py RelayMeasurementDataset.csv --rounds 3` (API) | extended knowledge base, proposal audit |
| 3 | `python eval_extended.py RelayMeasurementDataset.csv` | sealed-split result (Table V) |
| 4 | `python llm_stage.py RelayMeasurementDataset.csv --mode sample --n 300` (API) | constrained-classifier outputs |
| 5 | `python score_llm.py llm_qwen.jsonl` | faithfulness metrics |
| 6 | `python compare_systems.py RelayMeasurementDataset.csv llm_qwen.jsonl` | system comparison (Table III) |
| 7 | `python compare_llms.py RelayMeasurementDataset.csv` (API) | proposing-model comparison (Table VII) |
| 8 | `python seed_variance.py RelayMeasurementDataset.csv --repeats 3` (API) | run-to-run variance (Table VIII) |
| 9 | `python measure_latency.py RelayMeasurementDataset.csv` | latency (Table VI) |
| 10 | `python adaptive_adversary.py RelayMeasurementDataset.csv` | evasion results |
| 11 | `python edge_footprint.py RelayMeasurementDataset.csv` | edge deployability figures |
| 12 | `python make_figures.py` | all paper figures |

## A note on reproducibility of the API steps

Steps marked (API) are **not bit-reproducible**. Repeated runs of the same
model with identical prompts and temperature 0 produce different accepted
signatures, because the serving infrastructure is non-deterministic. This is
measured and reported in the paper. The frozen knowledge bases and raw model
responses included here are the exact artifacts behind the reported numbers,
so the offline steps reproduce the paper's results exactly even though a fresh
proposal run would not.

## Included result artifacts

- `calibrated_kb_tcdr.json`, `calibrated_kb_extended.json` — the 6- and
  9-signature knowledge bases
- `signature_weights*.json` — lift weights used for voting
- `proposal_audit.json` — every LLM proposal with its accept/reject verdict
  and measured statistics
- `tcdr_test_indices.npy` — the frozen test split, so the sealed evaluation is
  reproducible exactly
- `llm_qwen.jsonl` — raw model responses, so faithfulness rates can be
  recomputed without re-querying any API
- `raw_proposal_round*.txt` — unparsed model output from the proposal rounds
- remaining `*.json` — latency, evasion, variance and footprint results

## Requirements

Python 3.10+, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, and `openai`
(for the API steps only).

## Licence

Code released under the MIT Licence. The dataset is governed by its own terms;
see above.
"""


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(f"{OUT}/code")
    os.makedirs(f"{OUT}/results")

    import glob
    results = list(RESULTS)
    for g in GLOBS:
        results.extend(sorted(glob.glob(g)))

    copied, missing, secrets = [], [], []

    for name, dest in [(c, "code") for c in CODE] + [(r, "results") for r in results]:
        if name in FORBIDDEN:
            continue
        if not os.path.exists(name):
            missing.append(name)
            continue
        hits = scan(name)
        if hits:
            secrets.append((name, hits))
            continue
        shutil.copy2(name, f"{OUT}/{dest}/{name}")
        copied.append(f"{dest}/{name}")

    open(f"{OUT}/README.md", "w", encoding="utf-8").write(README)
    copied.append("README.md")

    print(f"Collected {len(copied)} files into {OUT}/")
    if missing:
        print(f"\nNot found (fine if you did not run that step):")
        for m in missing:
            print(f"   {m}")

    if secrets:
        print("\n" + "!" * 64)
        print("REFUSING TO PACKAGE — key-shaped strings found:")
        for name, hits in secrets:
            for kind, masked in hits:
                print(f"   {name}: {kind}  {masked}")
        print("Remove the key from these files, then re-run.")
        print("!" * 64)
        sys.exit(1)

    # Final sweep over everything actually staged.
    flagged = []
    for root, _, files in os.walk(OUT):
        for f in files:
            p = os.path.join(root, f)
            if scan(p):
                flagged.append(p)
    if flagged:
        print("\nREFUSING TO ZIP — secrets found in staged files:", flagged)
        sys.exit(1)

    zpath = "gridcopilot_artifact.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(OUT):
            for f in files:
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, "."))

    size = os.path.getsize(zpath)
    digest = hashlib.sha256(open(zpath, "rb").read()).hexdigest()
    print(f"\nWrote {zpath}  ({size/1024:.0f} KiB)")
    print(f"SHA-256: {digest}")
    print("\nNo secrets detected. The dataset is excluded by design.")
    print("Upload this zip to Zenodo, then paste the DOI back.")


if __name__ == "__main__":
    main()
