"""
BERTScore Evaluation Script (High-Quality Config)

Computes BERTScore (Precision, Recall, F1) between agent responses
and expert ground-truth answers using the best available settings:

  Model  : roberta-large  (recommended for English, paper default)
  Layers : 17             (optimal for roberta-large per BERTScore paper)
  Baseline rescaling: True — spreads scores into a more discriminative range
  Device : CUDA if available (RTX 5060 Ti), else CPU

Run with the torch virtual environment:
  C:\\Users\\23999\\.conda\\envs\\torch\\python.exe experiments/evaluate_bertscore.py <results.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime

# ── Model config ──────────────────────────────────────────────────────────────
# roberta-large + layer 17 is the top English configuration per the BERTScore
# paper (Zhang et al., 2020). With rescale_with_baseline the F1 values become
# more human-interpretable and better separated between good/poor responses.
BERT_MODEL  = "roberta-large"
NUM_LAYERS  = 17       # optimal layer for roberta-large
RESCALE     = True     # rescale against Common Crawl baseline → more spread
BATCH_SIZE  = 32       # reduce to 16 if you hit GPU OOM for very long texts


def load_results(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  CUDA device : {name}  ({vram:.1f} GB)")
            return "cuda"
    except Exception:
        pass
    print("  Device      : CPU (no CUDA)")
    return "cpu"


def compute_bertscore_batch(
    candidates: list[str],
    references: list[str],
    model: str,
    num_layers: int,
    rescale: bool,
    device: str,
) -> tuple[list[float], list[float], list[float]]:
    """Return P, R, F1 lists for the full batch."""
    from bert_score import score as bs_score

    P, R, F = bs_score(
        candidates,
        references,
        model_type=model,
        num_layers=num_layers,
        rescale_with_baseline=rescale,
        lang="en",
        device=device,
        batch_size=BATCH_SIZE,
        verbose=True,          # shows progress bar
    )
    return P.tolist(), R.tolist(), F.tolist()


def evaluate_bertscore(results_file: str, output_file: str = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print(f"BERTScore Evaluation  [{timestamp}]")
    print(f"  Model       : {BERT_MODEL} (layer {NUM_LAYERS})")
    print(f"  Rescale     : {RESCALE}")
    print(f"  Batch size  : {BATCH_SIZE}")
    device = detect_device()
    print("=" * 70)

    results = load_results(results_file)
    print(f"\nLoaded {len(results)} results from: {results_file}")

    # Collect valid (response, ground_truth) pairs
    valid_idx:  list[int] = []
    candidates: list[str] = []
    references: list[str] = []

    for i, r in enumerate(results):
        resp  = (r.get("response") or "").strip()
        gt    = (r.get("ground_truth") or "").strip()
        error = r.get("error") and not resp
        if resp and gt and not error:
            valid_idx.append(i)
            candidates.append(resp)
            references.append(gt)
        else:
            print(f"  [skip] {r.get('query_id','?')} — missing response or ground truth")

    print(f"\nComputing BERTScore for {len(valid_idx)}/{len(results)} items...\n")

    P_list, R_list, F_list = compute_bertscore_batch(
        candidates, references, BERT_MODEL, NUM_LAYERS, RESCALE, device
    )

    # Enrich original results
    score_map = {
        idx: (P_list[k], R_list[k], F_list[k])
        for k, idx in enumerate(valid_idx)
    }

    enriched = []
    for i, r in enumerate(results):
        entry = dict(r)
        if i in score_map:
            p, rec, f1 = score_map[i]
            entry["bertscore"] = {
                "precision": round(p,   6),
                "recall":    round(rec, 6),
                "f1":        round(f1,  6),
                "model":     BERT_MODEL,
                "num_layers": NUM_LAYERS,
                "rescaled":  RESCALE,
            }
        else:
            entry["bertscore"] = None
        enriched.append(entry)

    # Determine output path
    if not output_file:
        out_dir     = Path(results_file).parent
        stem        = Path(results_file).stem
        output_file = str(out_dir / f"{stem}_bertscore.json")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)

    # Summary
    ctrl_f1 = [score_map[i][2] for i in score_map
               if results[i].get("group") == "Control"]
    exp_f1  = [score_map[i][2] for i in score_map
               if results[i].get("group") == "Experimental"]

    def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0

    c_f1 = avg(ctrl_f1)
    e_f1 = avg(exp_f1)
    delta = (e_f1 - c_f1) / c_f1 * 100 if c_f1 != 0 else 0.0

    print(f"\n{'='*70}")
    print(f"Done!  {len(valid_idx)} items scored.")
    print(f"  Control F1      : {c_f1:.4f}  (n={len(ctrl_f1)})")
    print(f"  Experimental F1 : {e_f1:.4f}  (n={len(exp_f1)})")
    print(f"  Δ F1            : {delta:+.2f}%")
    print(f"\nSaved to : {output_file}")
    print(f"{'='*70}")

    return output_file


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  C:\\Users\\23999\\.conda\\envs\\torch\\python.exe "
              "experiments/evaluate_bertscore.py <results.json>")
        sys.exit(1)
    out = evaluate_bertscore(sys.argv[1])
    print(f"\nNext: python experiments/generate_report_combined.py "
          f"experiments/report/<tfidf_eval>.json {out}")


if __name__ == "__main__":
    main()
