# CounselChat RAG Experiment - Quick Start Guide

## Overview

This experiment evaluates the impact of integrating CounselChat data into the SmartStress Agent's RAG system through an automated A/B test.

## Prerequisites

1. TiDB with RAG documents ingested (run `python ingest_counselchat_tidb.py` to add remaining documents)
2. Python environment with all dependencies installed
3. `GOOGLE_API_KEY` configured in `.env`

## Experiment Workflow

### Step 1: Run A/B Test

Execute test queries through both control and experimental groups:

```bash
python experiments/run_ab_test.py
```

**What it does:**
- Loads 25 test queries from `experiments/test_queries.json`
- Runs each query through:
  - **Control Group:** No RAG enhancement
  - **Experimental Group:** With CounselChat RAG (k=3)
- Saves raw results to `experiments/ab_test_results_TIMESTAMP.json`

**Expected duration:** ~30-60 minutes (depends on agent response time)

### Step 2: Evaluate Results

Use LLM-as-a-judge to score all responses:

```bash
python experiments/evaluate_results.py experiments/ab_test_results_TIMESTAMP.json
```

**What it does:**
- Loads A/B test results
- Evaluates each response using Gemini as judge
- Scores on 4 metrics (1-5 scale):
  - Groundedness
  - Stressor Identification
  - Safety Compliance
  - Response Quality
- Saves evaluated results to `experiments/ab_test_results_TIMESTAMP_evaluated.json`

**Expected duration:** ~20-40 minutes (25 queries × 2 groups = 50 evaluations)

### Step 3: Generate Report

Create a comprehensive comparison report:

```bash
python experiments/generate_report.py experiments/ab_test_results_TIMESTAMP_evaluated.json
```

**What it does:**
- Calculates statistics for both groups
- Compares performance metrics
- Generates findings and recommendations
- Creates Markdown report: `experiments/ab_test_results_TIMESTAMP_report.md`

## Files Created

| File                               | Description                                      |
| ---------------------------------- | ------------------------------------------------ |
| `test_queries.json`                | 25 test queries across diverse stress categories |
| `ab_test_config.py`                | Test configuration and evaluation metrics        |
| `run_ab_test.py`                   | A/B test runner                                  |
| `evaluate_results.py`              | LLM-as-a-judge evaluator                         |
| `generate_report.py`               | Report generator                                 |
| `ab_test_results_*.json`           | Raw test results                                 |
| `ab_test_results_*_evaluated.json` | Evaluated results with scores                    |
| `ab_test_results_*_report.md`      | Final comparison report                          |

## Key Configuration

### Test Groups

**Control (Group A):**
- No RAG enhancement
- Baseline agent performance

**Experimental (Group B):**
- CounselChat RAG enabled
- Retrieves k=3 most similar documents
- Filters by tags: `["psychoeducation", "counselchat"]`

### Evaluation Metrics

Each response is scored 1-5 on:
1. **Groundedness:** Evidence-based, grounded in professional knowledge
2. **Stressor Identification:** Accurately identifies user's stressors
3. **Safety Compliance:** Safe, appropriate, ethical advice
4. **Response Quality:** Overall helpfulness and empathy

## Troubleshooting

### API Rate Limits

If you encounter rate limit errors during evaluation:
- The scripts will continue with errors logged
- Consider adding delays between evaluation calls
- Results with errors will be marked in the output

### Missing Documents in TiDB

If TiDB doesn't have enough documents (should be ~868):
```bash
python ingest_counselchat_tidb.py
```
The script will skip already-ingested documents and only process new ones.

## Next Steps

After generating the report:
1. Review the findings in `*_report.md`
2. Analyze which categories benefited most from RAG
3. Consider adjusting RAG parameters (k value, retrieval method)
4. If results are positive, integrate RAG into production agent
