"""Regenerate comparison.png from the judge-independent eval metrics.

The LLM-judged grounding score is noisy when a small model is used as the
auditor, so this chart is built only from deterministic, reproducible signals:
abstention on unanswerable questions, and overall status accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "eval" / "results"


def metrics(scores: list[dict]) -> dict:
    unans = [s for s in scores if s["expected_status"] == ["abstain"]]
    ans = [s for s in scores if s["expected_status"] == ["success"]]
    hallucinated_unans = sum(1 for s in unans if s["answered"])
    correct_abst = sum(1 for s in unans if not s["answered"])
    status_ok = sum(1 for s in scores if s["status_correct"])
    return {
        "halluc_unans": hallucinated_unans / (len(unans) or 1),
        "abst_acc": correct_abst / (len(unans) or 1),
        "status_acc": status_ok / (len(scores) or 1),
    }


def main() -> None:
    d = json.loads((RES / "results.json").read_text())
    b = metrics(d["baseline"]["scores"])
    a = metrics(d["agent"]["scores"])

    labels = ["Hallucinated on\nunanswerable Qs\n(lower is better)",
              "Correct\nabstentions", "Status\naccuracy"]
    bvals = [b["halluc_unans"], b["abst_acc"], b["status_acc"]]
    avals = [a["halluc_unans"], a["abst_acc"], a["status_acc"]]

    x = range(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - w / 2 for i in x], [v * 100 for v in bvals], w,
           label="Baseline RAG", color="#94a3b8")
    ax.bar([i + w / 2 for i in x], [v * 100 for v in avals], w,
           label="VeriRAG (self-correcting)", color="#7c3aed")
    for i, (bv, av) in enumerate(zip(bvals, avals)):
        ax.text(i - w / 2, bv * 100 + 1.5, f"{bv*100:.0f}%", ha="center", fontsize=10)
        ax.text(i + w / 2, av * 100 + 1.5, f"{av*100:.0f}%", ha="center", fontsize=10)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("%")
    ax.set_ylim(0, 115)
    ax.set_title("VeriRAG - impact of the self-correction layer",
                 fontsize=13, weight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RES / "comparison.png", dpi=160)
    print("wrote", RES / "comparison.png")
    print("baseline", b)
    print("agent", a)


if __name__ == "__main__":
    main()
