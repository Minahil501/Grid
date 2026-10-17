# GridCopilot — code and result artifacts

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
