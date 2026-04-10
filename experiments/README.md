# CounselChat RAG Experiment - Quick Start Guide

## Overview

This experiment measures the impact of CounselChat RAG on SmartStress responses
using an A/B test:
- Control: `use_rag=False`
- Experimental: `use_rag=True`

The current evaluation pipeline is metric-based (not LLM-judge):
- TF-IDF cosine similarity (`evaluate_results.py`)
- Optional BERTScore (`evaluate_bertscore.py`)

## Prerequisites

1. TiDB has RAG documents ingested (for example: `python ingest_counselchat_tidb.py`)
2. Python environment with dependencies installed (`pip install -r requirements.txt`)
3. `GOOGLE_API_KEY` configured in `.env`

For BERTScore step, prepare a PyTorch-capable env (CPU or CUDA) and install:
- `bert-score`
- `torch`

## Workflow

### Step 1: Run A/B Test

```bash
python experiments/run_ab_test.py
```

What it does:
- Loads queries from `experiments/test_queries.json`
- Runs each query in both control/experimental groups
- Saves raw output to `experiments/report/ab_test_results_<timestamp>.json`

### Step 2A: Evaluate with TF-IDF (default)

```bash
python experiments/evaluate_results.py experiments/report/ab_test_results_<timestamp>.json
```

What it does:
- Computes TF-IDF cosine similarity between response and ground truth
- Produces `experiments/report/ab_test_results_<timestamp>_evaluated.json`

### Step 2B: Evaluate with BERTScore (optional)

```bash
python experiments/evaluate_bertscore.py experiments/report/ab_test_results_<timestamp>.json
```

What it does:
- Computes Precision / Recall / F1 via BERTScore
- Produces `experiments/report/ab_test_results_<timestamp>_bertscore.json`

### Step 3A: Generate TF-IDF Report

```bash
python experiments/generate_report.py experiments/report/ab_test_results_<timestamp>_evaluated.json
```

### Step 3B: Generate Combined Report (TF-IDF + BERTScore)

```bash
python experiments/generate_report_combined.py experiments/report/ab_test_results_<timestamp>_evaluated.json experiments/report/ab_test_results_<timestamp>_bertscore.json
```

## Files and Roles

| File | Role |
|---|---|
| `test_queries.json` | Query set |
| `ab_test_config.py` | Group config and metric config |
| `run_ab_test.py` | A/B runner |
| `evaluate_results.py` | TF-IDF evaluator |
| `evaluate_bertscore.py` | BERTScore evaluator (optional) |
| `generate_report.py` | TF-IDF report |
| `generate_report_combined.py` | Combined metrics report |

## Key Metrics

1. TF-IDF cosine similarity (lexical alignment)
2. BERTScore F1/Precision/Recall (semantic alignment)
3. Group deltas and significance from report scripts

## Troubleshooting

### TiDB documents missing

If retrieval quality is low because docs were not ingested:

```bash
python ingest_counselchat_tidb.py
```

### BERTScore runtime issues

- If GPU OOM occurs, reduce batch size in `evaluate_bertscore.py`.
- If no CUDA is available, script falls back to CPU.
