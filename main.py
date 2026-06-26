"""Bot-or-Not -- end-to-end pipeline.

Runs the whole thing with one command:

    python main.py

Steps: parse raw datasets -> engineer features -> benchmark models with
stratified cross-validation -> print a report -> write the feature table and
the web showcase data.

Flags:
    --no-export   skip writing JSON/CSV artifacts (just print the report)
    --quiet       suppress the per-model table
"""
from __future__ import annotations

import argparse
import warnings

from src import export, model
from src.dataset import build_feature_table, save_feature_table
from src.parser import load_users

# scikit-learn emits convergence/feature-name chatter that would bury the
# report; we silence it deliberately rather than globally at import time.
warnings.filterwarnings("ignore")

FEATURES_CSV = "data/features.csv"
RESULTS_JSON = "docs/data/results.json"
RESULTS_JS = "docs/data/results.js"


def _print_report(results: model.Results, n_posts: int) -> None:
    best = results.best
    base = model.BASELINE

    print("\n" + "=" * 64)
    print("  BOT-OR-NOT  --  detection report")
    print("=" * 64)
    print(
        f"  Dataset : {results.n_users} users  "
        f"({results.n_bots} bots / {results.n_users - results.n_bots} humans), "
        f"{n_posts:,} posts"
    )
    print(f"  Features: {results.n_features} behavioural signals per user")

    print("\n  Model comparison (5-fold out-of-fold):")
    print(f"  {'model':26} {'ROC-AUC':>8} {'PR-AUC':>8} {'bot-R':>7} {'bot-F1':>7}")
    print("  " + "-" * 58)
    for ev in results.evaluations.values():
        marker = " *" if ev.name == results.selected else "  "
        print(
            f"  {ev.name:26} {ev.roc_auc:8.3f} {ev.pr_auc:8.3f} "
            f"{ev.bot_recall:7.3f} {ev.bot_f1:7.3f}{marker}"
        )
    print(f"\n  Selected: {results.selected}  (* )")

    print("\n  Improvement over original pipeline:")
    print(f"  {'metric':16} {'original':>10} {'improved':>10} {'delta':>8}")
    print("  " + "-" * 46)
    for key, label in [
        ("accuracy", "accuracy"),
        ("bot_recall", "bot recall"),
        ("bot_f1", "bot F1"),
    ]:
        b = base[key]
        n = getattr(best, key)
        print(f"  {label:16} {b:10.2f} {n:10.2f} {n - b:+8.2f}")

    print("\n  Top signals:")
    for name, score in results.importances[:6]:
        print(f"    {score:6.3f}  {name}")
    print("=" * 64 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Bot-or-Not pipeline.")
    ap.add_argument("--no-export", action="store_true", help="don't write artifacts")
    ap.add_argument("--quiet", action="store_true", help="suppress the report")
    args = ap.parse_args()

    print("Loading datasets and engineering features ...")
    n_posts = sum(len(u.posts) for u in load_users())
    table = build_feature_table()

    print("Benchmarking models with stratified cross-validation ...")
    results = model.run(table)

    if not args.quiet:
        _print_report(results, n_posts)

    if not args.no_export:
        save_feature_table(table, FEATURES_CSV)
        payload = export.build_payload(results, n_posts)
        export.write_payload(payload, RESULTS_JSON)
        export.write_js_payload(payload, RESULTS_JS)
        print(f"Wrote {FEATURES_CSV}, {RESULTS_JSON} and {RESULTS_JS}")


if __name__ == "__main__":
    main()
