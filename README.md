# contract-eval
### LLM Evaluation Framework for Legal Contract Review

An end-to-end evaluation system for AI-powered contract clause extraction — built to demonstrate the core Data Scientist responsibilities at Crosby AI.

## What it does

Benchmarks three LLM extractors on 20 synthetic contracts (NDAs, SaaS agreements, MSAs) across 14 clause types with ground-truth annotations.

| Module | Purpose |
|--------|---------|
| `pipeline.py` | Generate contracts, simulate LLM predictions, compute all metrics |
| `dashboard.py` | Interactive Streamlit evaluation dashboard |

## Metrics

- **Macro F1** — clause-level precision/recall across all types
- **Risk-Weighted F1** — penalises errors on high-risk clauses 3× (missed indemnification > missed governing law)
- **Red Flag F1** — precision/recall for detecting specific risky patterns
- **ECE (Expected Calibration Error)** — confidence calibration for human review routing

## Models benchmarked

| Model | Macro F1 | Risk-W F1 | Red Flag F1 |
|-------|----------|-----------|-------------|
| `gpt4_zeroshot`  | ~0.71 | ~0.72 | ~0.63 |
| `claude_fewshot` | ~0.83 | ~0.84 | ~0.77 |
| `ensemble_v1`    | ~0.90 | ~0.91 | ~0.87 |

## Run

```bash
pip install -r requirements.txt
python pipeline.py
streamlit run dashboard.py
```

## Design rationale

Legal contract review has an asymmetric error cost: a **false negative** (missed high-risk clause) exposes a client to unreviewed legal risk. The risk-weighted F1 metric directly encodes this — high-risk clause errors (limitation of liability, indemnification, IP ownership, data privacy) are penalised 3× more than low-risk clause errors. This is the same principle that would drive labeling strategy and model iteration at Crosby.
