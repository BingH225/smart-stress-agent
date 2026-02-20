"""
Combined Report Generator — TF-IDF Similarity + BERTScore

Reads two evaluated result files (TF-IDF and BERTScore) and produces one
unified Markdown report with:
  - Per-group statistics for both metrics
  - Welch's independent-samples t-test for each metric
  - Per-category breakdown
  - Key findings and recommendations

Usage:
  python experiments/generate_report_combined.py \\
      experiments/report/ab_test_results_*_evaluated.json \\
      experiments/report/ab_test_results_*_bertscore.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple


# ── helpers ───────────────────────────────────────────────────────────────────

def load(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_scores(results: List[Dict], group: str, key_fn) -> List[float]:
    """Extract a list of scores for the given group using key_fn(result) -> float|None."""
    out = []
    for r in results:
        if r.get("group") != group:
            continue
        v = key_fn(r)
        if isinstance(v, float) and v > 0:
            out.append(v)
    return out


def stats(values: List[float]) -> Dict[str, float]:
    n = len(values)
    return {
        "mean":  round(mean(values), 4)  if n else 0.0,
        "stdev": round(stdev(values), 4) if n > 1 else 0.0,
        "min":   round(min(values), 4)   if n else 0.0,
        "max":   round(max(values), 4)   if n else 0.0,
        "n":     n,
    }


def welch_t(a: List[float], b: List[float]) -> Tuple[Optional[float], Optional[float], str]:
    """Welch's independent t-test. Returns (t, p, interpretation)."""
    try:
        from scipy import stats as sc_stats
        if len(a) < 2 or len(b) < 2:
            return None, None, "insufficient data"
        t, p = sc_stats.ttest_ind(a, b, equal_var=False)
        if p < 0.001:   sig = "highly significant (p < 0.001)"
        elif p < 0.01:  sig = "significant (p < 0.01)"
        elif p < 0.05:  sig = "significant (p < 0.05)"
        elif p < 0.10:  sig = "marginally significant (p < 0.10)"
        else:            sig = "not significant (p ≥ 0.10)"
        return float(t), float(p), sig
    except ImportError:
        return None, None, "scipy not installed"
    except Exception as e:
        return None, None, str(e)


def pct_change(base: float, new: float) -> str:
    if base == 0:
        return "N/A"
    v = (new - base) / base * 100
    return f"+{v:.1f}%" if v >= 0 else f"{v:.1f}%"


def category_means(results: List[Dict], group: str, key_fn) -> Dict[str, float]:
    cat_data: Dict[str, List[float]] = {}
    for r in results:
        if r.get("group") != group:
            continue
        cat = r.get("category", "unknown")
        v   = key_fn(r)
        if isinstance(v, float):
            cat_data.setdefault(cat, []).append(v)
    return {c: round(mean(vs), 4) for c, vs in sorted(cat_data.items())}


# ── section builders ──────────────────────────────────────────────────────────

def metric_section(
    lines: List[str],
    title: str,
    metric_label: str,
    ctrl_scores: List[float],
    exp_scores: List[float],
    ctrl_cats: Dict[str, float],
    exp_cats: Dict[str, float],
):
    """Append a full metric comparison section (stats + t-test + categories)."""
    cs = stats(ctrl_scores)
    es = stats(exp_scores)
    t, p, sig = welch_t(ctrl_scores, exp_scores)

    lines.append(f"## {title}\n\n")

    # Summary table
    lines += [
        f"| | n | Mean {metric_label} | Std Dev | Min | Max |\n",
        f"|---|---|---|---|---|---|\n",
        f"| **Control (No RAG)** | {cs['n']} | {cs['mean']:.4f} | {cs['stdev']:.4f} | {cs['min']:.4f} | {cs['max']:.4f} |\n",
        f"| **Experimental (RAG k=3)** | {es['n']} | {es['mean']:.4f} | {es['stdev']:.4f} | {es['min']:.4f} | {es['max']:.4f} |\n",
        "\n",
    ]

    delta_pct = pct_change(cs["mean"], es["mean"])
    lines.append(f"**Mean change (RAG vs No-RAG):** {delta_pct}\n\n")

    # T-test
    if t is not None:
        sym = "✅" if p < 0.05 else "❌"
        lines += [
            f"### Statistical Test (Welch's t-test)\n\n",
            f"| t-statistic | p-value | Result |\n",
            f"|---|---|---|\n",
            f"| {t:.4f} | {p:.4f} | {sym} {sig} |\n",
            "\n",
        ]

    # Per-category
    all_cats = sorted(set(list(ctrl_cats) + list(exp_cats)))
    if all_cats:
        lines += [
            f"### Per-Category {metric_label}\n\n",
            "| Category | Control | Experimental | Δ |\n",
            "|---|---|---|---|\n",
        ]
        for cat in all_cats:
            c = ctrl_cats.get(cat, 0.0)
            e = exp_cats.get(cat, 0.0)
            d = e - c
            d_str = f"+{d:.4f}" if d >= 0 else f"{d:.4f}"
            lines.append(f"| {cat} | {c:.4f} | {e:.4f} | {d_str} |\n")
        lines.append("\n")

    lines.append("---\n\n")


# ── main report ───────────────────────────────────────────────────────────────

def generate_combined_report(
    tfidf_file: str,
    bertscore_file: str,
    output_file: str = None,
) -> str:
    print("=" * 70)
    print("Generating Combined Report (TF-IDF + BERTScore)")
    print("=" * 70)

    tfidf_results = load(tfidf_file)
    bs_results    = load(bertscore_file)

    # Index bertscore results by (query_id, group) composite key
    # — same query_id can appear in both Control and Experimental groups
    bs_map = {(r.get("query_id"), r.get("group")): r for r in bs_results}

    # ── Extract all score vectors ─────────────────────────────────────────────
    def tfidf_score(r):
        ev = r.get("evaluation") or {}
        return ev.get("ground_truth_similarity")

    def _bs_entry(r):
        return bs_map.get((r.get("query_id"), r.get("group")), {}).get("bertscore") or {}

    def bs_f1(r):
        v = _bs_entry(r).get("f1")
        return float(v) if isinstance(v, (int, float)) else None

    def bs_precision(r):
        v = _bs_entry(r).get("precision")
        return float(v) if isinstance(v, (int, float)) else None

    def bs_recall(r):
        v = _bs_entry(r).get("recall")
        return float(v) if isinstance(v, (int, float)) else None


    ctrl_tfidf = group_scores(tfidf_results, "Control",      tfidf_score)
    exp_tfidf  = group_scores(tfidf_results, "Experimental", tfidf_score)
    ctrl_f1    = group_scores(tfidf_results, "Control",      bs_f1)
    exp_f1     = group_scores(tfidf_results, "Experimental", bs_f1)
    ctrl_prec  = group_scores(tfidf_results, "Control",      bs_precision)
    exp_prec   = group_scores(tfidf_results, "Experimental", bs_precision)
    ctrl_rec   = group_scores(tfidf_results, "Control",      bs_recall)
    exp_rec    = group_scores(tfidf_results, "Experimental", bs_recall)

    # Category breakdowns
    ctrl_cats_tfidf = category_means(tfidf_results, "Control",      tfidf_score)
    exp_cats_tfidf  = category_means(tfidf_results, "Experimental", tfidf_score)
    ctrl_cats_f1    = category_means(tfidf_results, "Control",      bs_f1)
    exp_cats_f1     = category_means(tfidf_results, "Experimental", bs_f1)

    # ── Build report ──────────────────────────────────────────────────────────
    lines: List[str] = []

    # Header
    lines += [
        "# CounselChat RAG Enhancement — Combined A/B Test Report\n\n",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        "---\n\n",
    ]

    # Executive Summary
    ctrl_n   = sum(1 for r in tfidf_results if r.get("group") == "Control")
    exp_n    = sum(1 for r in tfidf_results if r.get("group") == "Experimental")
    bs_model = (bs_results[0].get("bertscore") or {}).get("model", "distilbert-base-uncased") \
               if bs_results else "distilbert-base-uncased"

    cs_t = stats(ctrl_tfidf); es_t = stats(exp_tfidf)
    cs_f = stats(ctrl_f1);    es_f = stats(exp_f1)

    lines += [
        "## Executive Summary\n\n",
        f"Comparing SmartStress Agent **with** and **without** CounselChat RAG,\n",
        f"evaluated using two complementary metrics:\n\n",
        f"| Metric | Method | Control | Experimental | Δ |\n",
        f"|---|---|---|---|---|\n",
        f"| TF-IDF Cosine Similarity | Lexical overlap | {cs_t['mean']:.4f} | {es_t['mean']:.4f} | {pct_change(cs_t['mean'], es_t['mean'])} |\n",
        f"| BERTScore F1 | Semantic (contextual) | {cs_f['mean']:.4f} | {es_f['mean']:.4f} | {pct_change(cs_f['mean'], es_f['mean'])} |\n",
        f"\n",
        f"> **BERTScore model:** `{bs_model}`  \n",
        f"> **Queries:** Control = {ctrl_n}, Experimental = {exp_n}\n\n",
        "---\n\n",
    ]

    # Metric 1: TF-IDF
    metric_section(
        lines,
        title="TF-IDF Cosine Similarity",
        metric_label="(0–1)",
        ctrl_scores=ctrl_tfidf,
        exp_scores=exp_tfidf,
        ctrl_cats=ctrl_cats_tfidf,
        exp_cats=exp_cats_tfidf,
    )

    # Metric 2: BERTScore F1
    metric_section(
        lines,
        title="BERTScore F1",
        metric_label="(0–1)",
        ctrl_scores=ctrl_f1,
        exp_scores=exp_f1,
        ctrl_cats=ctrl_cats_f1,
        exp_cats=exp_cats_f1,
    )

    # Metric 3: BERTScore P & R (compact)
    cs_p = stats(ctrl_prec); es_p = stats(exp_prec)
    cs_r = stats(ctrl_rec);  es_r = stats(exp_rec)
    lines += [
        "## BERTScore Precision & Recall\n\n",
        "| | Control P | Exp P | Δ P | Control R | Exp R | Δ R |\n",
        "|---|---|---|---|---|---|---|\n",
        f"| Mean | {cs_p['mean']:.4f} | {es_p['mean']:.4f} | {pct_change(cs_p['mean'], es_p['mean'])} "
        f"| {cs_r['mean']:.4f} | {es_r['mean']:.4f} | {pct_change(cs_r['mean'], es_r['mean'])} |\n",
        "\n---\n\n",
    ]

    # Key Findings
    t_tfidf, p_tfidf, sig_tfidf = welch_t(ctrl_tfidf, exp_tfidf)
    t_f1,    p_f1,    sig_f1    = welch_t(ctrl_f1,    exp_f1)

    lines += ["## Key Findings\n\n"]

    for label, cs, es, t_v, p_v, sig in [
        ("TF-IDF Similarity", cs_t, es_t, t_tfidf, p_tfidf, sig_tfidf),
        ("BERTScore F1",      cs_f, es_f, t_f1,    p_f1,    sig_f1),
    ]:
        delta = pct_change(cs["mean"], es["mean"])
        direction = "📈 improves" if es["mean"] > cs["mean"] else "📉 decreases"
        sym = "✅" if (p_v is not None and p_v < 0.05) else "❌"
        lines += [
            f"### {label}\n",
            f"- RAG {direction} mean score by **{delta}** ({cs['mean']:.4f} → {es['mean']:.4f})\n",
        ]
        if t_v is not None:
            lines.append(f"- {sym} t = {t_v:.4f}, p = {p_v:.4f} — **{sig}**\n")
        lines.append("\n")

    lines.append("---\n\n")

    # Recommendations
    f1_sig   = p_f1 is not None and p_f1 < 0.05
    f1_up    = es_f["mean"] > cs_f["mean"]
    lines += ["## Recommendations\n\n"]
    if f1_sig and f1_up:
        lines += [
            "1. **Deploy RAG:** BERTScore improvement is statistically significant — "
            "evidence supports production deployment.\n",
            "2. **Monitor:** Track both TF-IDF and BERTScore in production.\n",
            "3. **Expand Knowledge Base:** More high-quality counseling resources may "
            "further improve semantic alignment.\n",
        ]
    else:
        lines += [
            "1. **Tune Retrieval:** BERTScore improvement is not yet statistically "
            "significant. Consider adjusting k, similarity thresholds, or prompt integration.\n",
            "2. **Improve Context Integration:** Review how RAG context is inserted "
            "into the agent's system prompt.\n",
            "3. **Expand Knowledge Base:** Additional or higher-quality counseling "
            "documents may improve coverage.\n",
        ]
    lines.append("\n---\n\n")

    # Methodology
    lines += [
        "## Methodology\n\n",
        "| | Detail |\n",
        "|---|---|\n",
        "| TF-IDF | Cosine similarity using unigrams + bigrams, stop-word removal |\n",
        f"| BERTScore | Contextual token embedding similarity via `{bs_model}` |\n",
        "| Statistical Test | Welch's independent-samples t-test (two-tailed, unequal variance) |\n",
        "| RAG config | k=3 retrieved documents, CounselChat dataset |\n",
        "| Test data | `counselchat-data.csv` (held-out from RAG ingestion) |\n",
    ]

    # Save
    if not output_file:
        out_dir = Path(tfidf_file).parent
        output_file = str(out_dir / f"combined_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_text("".join(lines), encoding="utf-8")

    print(f"\n{'='*70}")
    print(f"Combined report saved to: {output_file}")
    print(f"  TF-IDF  — Control: {cs_t['mean']:.4f} | Experimental: {es_t['mean']:.4f} | Δ {pct_change(cs_t['mean'], es_t['mean'])}")
    print(f"  BERTScore F1 — Control: {cs_f['mean']:.4f} | Experimental: {es_f['mean']:.4f} | Δ {pct_change(cs_f['mean'], es_f['mean'])}")
    print(f"{'='*70}")

    return output_file


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python experiments/generate_report_combined.py <tfidf_evaluated.json> <bertscore.json>")
        sys.exit(1)
    generate_combined_report(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
