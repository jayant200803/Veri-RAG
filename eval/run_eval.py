"""Run the headline experiment: baseline RAG vs. the VeriRAG agent.

    python eval/run_eval.py            # both pipelines, full golden set
    python eval/run_eval.py --only agent
    python eval/run_eval.py --no-chart

Outputs:
    eval/results/results.json   raw per-question scores + aggregates
    eval/results/comparison.png bar chart for the README and final slide
    stdout                      the table you put on your slide
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.graph.build import run_agent          # noqa: E402
from app.logging_conf import configure_logging  # noqa: E402
from eval.baseline_rag import run_baseline      # noqa: E402
from eval.scorer import aggregate, score_response  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
GOLDEN = Path(__file__).resolve().parents[1] / "data" / "golden" / "qa.yaml"


def load_golden() -> list[dict]:
    with GOLDEN.open() as f:
        return yaml.safe_load(f)["questions"]


def run_pipeline(name: str, runner, questions: list[dict]) -> dict:
    print(f"\n{'=' * 62}\n  {name}\n{'=' * 62}")
    scores = []

    for q in questions:
        print(f"  {q['id']} [{q['category']:<18}] {q['question'][:52]}...")
        try:
            result = runner(q["question"])
        except Exception as exc:
            print(f"       ERROR: {exc}")
            result = {"question": q["question"], "answer": f"ERROR: {exc}",
                      "status": "abstain", "sources": [], "confidence": 0.0}

        score = score_response(result, q)
        scores.append(score)

        flag = "HALLUCINATION" if score["hallucinated"] else "ok"
        print(f"       -> status={score['status']:<8} {flag}")

    return {"name": name, "scores": scores, "summary": aggregate(scores)}


def print_comparison(baseline: dict, agent: dict) -> None:
    b, a = baseline["summary"], agent["summary"]

    def pct(x):
        return f"{x * 100:.1f}%" if isinstance(x, (int, float)) else str(x)

    rows = [
        ("Hallucination rate", pct(b["hallucination_rate"]), pct(a["hallucination_rate"])),
        ("Correct abstentions", b["correct_abstentions"], a["correct_abstentions"]),
        ("Answerable questions answered", b["answerable_answered"], a["answerable_answered"]),
        ("Content accuracy", pct(b["content_accuracy"]), pct(a["content_accuracy"])),
        ("Status accuracy", pct(b["status_accuracy"]), pct(a["status_accuracy"])),
        ("Mean latency", f"{b['mean_latency_ms']}ms", f"{a['mean_latency_ms']}ms"),
    ]

    print(f"\n\n{'=' * 74}")
    print("  HEADLINE RESULT — before vs. after the self-correction layer")
    print("=" * 74)
    print(f"  {'Metric':<34}{'Baseline RAG':>18}{'VeriRAG':>18}")
    print("  " + "-" * 70)
    for label, bv, av in rows:
        print(f"  {label:<34}{bv:>18}{av:>18}")
    print("  " + "-" * 70)

    if b["hallucination_rate"] > 0:
        factor = b["hallucination_rate"] / max(a["hallucination_rate"], 1e-9)
        if a["hallucination_rate"] == 0:
            print(f"  Hallucinations eliminated entirely "
                  f"({b['hallucinations']} -> 0)")
        else:
            print(f"  {factor:.1f}x reduction in hallucination rate")
    print("=" * 74)


def make_chart(baseline: dict, agent: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping chart")
        return

    b, a = baseline["summary"], agent["summary"]
    metrics = ["Hallucination\nrate", "Abstention\naccuracy", "Content\naccuracy"]
    bvals = [b["hallucination_rate"], b["abstention_accuracy"], b["content_accuracy"]]
    avals = [a["hallucination_rate"], a["abstention_accuracy"], a["content_accuracy"]]

    x = range(len(metrics))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width / 2 for i in x], [v * 100 for v in bvals], width,
           label="Baseline RAG", color="#94a3b8")
    ax.bar([i + width / 2 for i in x], [v * 100 for v in avals], width,
           label="VeriRAG (self-correcting)", color="#7c3aed")

    for i, (bv, av) in enumerate(zip(bvals, avals)):
        ax.text(i - width / 2, bv * 100 + 1.5, f"{bv * 100:.0f}%",
                ha="center", fontsize=10)
        ax.text(i + width / 2, av * 100 + 1.5, f"{av * 100:.0f}%",
                ha="center", fontsize=10)

    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylabel("%")
    ax.set_ylim(0, 115)
    ax.set_title("VeriRAG — impact of the self-correction layer",
                 fontsize=13, weight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = RESULTS_DIR / "comparison.png"
    fig.savefig(out, dpi=160)
    print(f"\nChart written to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["baseline", "agent"])
    parser.add_argument("--no-chart", action="store_true")
    args = parser.parse_args()

    configure_logging("WARNING")   # keep the eval output readable
    questions = load_golden()
    print(f"Loaded {len(questions)} golden questions from {GOLDEN.name}")

    payload: dict = {}

    if args.only != "agent":
        payload["baseline"] = run_pipeline("BASELINE RAG (no self-correction)",
                                           run_baseline, questions)
    if args.only != "baseline":
        payload["agent"] = run_pipeline("VERIRAG AGENT (self-correcting)",
                                        run_agent, questions)

    if "baseline" in payload and "agent" in payload:
        print_comparison(payload["baseline"], payload["agent"])
        if not args.no_chart:
            make_chart(payload["baseline"], payload["agent"])

    out = RESULTS_DIR / "results.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"Raw results written to {out}")


if __name__ == "__main__":
    main()
