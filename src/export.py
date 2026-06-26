"""Serialize model results to a single JSON the static frontend consumes."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import model as model_mod
from .features import FEATURE_META, FEATURE_NAMES

# Compact set of raw features surfaced per-user in the interactive explorer.
EXPLORER_FEATURES = [
    "z_score",
    "post_count",
    "posts_per_hour",
    "avg_tweet_len",
    "std_time_gap",
    "url_ratio",
    "avg_hashtags",
    "night_post_ratio",
]


def build_payload(results: model_mod.Results, n_posts: int) -> dict[str, Any]:
    table = results.feature_table
    y = table["is_bot"].to_numpy()
    best = results.best

    # Per-user records: out-of-fold probability + a few interpretable features.
    users = []
    for (_, row), proba, pred in zip(table.iterrows(), best.proba, best.pred):
        users.append(
            {
                "username": row["username"],
                "dataset": row["dataset"],
                "is_bot": int(row["is_bot"]),
                "bot_probability": round(float(proba), 4),
                "predicted": int(pred),
                "features": {f: round(float(row[f]), 4) for f in EXPLORER_FEATURES},
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "dataset": {
            "n_users": results.n_users,
            "n_bots": results.n_bots,
            "n_humans": results.n_users - results.n_bots,
            "n_posts": n_posts,
            "n_features": results.n_features,
            "n_rounds": int(table["dataset"].nunique()),
            "bot_rate": round(results.n_bots / results.n_users, 4),
        },
        "baseline": model_mod.BASELINE,
        "selected_model": results.selected,
        "metrics": best.summary(),
        "model_comparison": [
            ev.summary() for ev in results.evaluations.values()
        ],
        "confusion_matrix": model_mod.confusion(y, best.pred),
        "curves": model_mod.curve_points(y, best.proba),
        "feature_importances": [
            {
                "name": name,
                "label": FEATURE_META.get(name, {}).get("label", name),
                "group": FEATURE_META.get(name, {}).get("group", ""),
                "desc": FEATURE_META.get(name, {}).get("desc", ""),
                "importance": round(score, 4),
            }
            for name, score in results.importances
        ],
        "feature_meta": [
            {"name": name, **FEATURE_META.get(name, {})} for name in FEATURE_NAMES
        ],
        "explorer_features": [
            {"name": name, **FEATURE_META.get(name, {})} for name in EXPLORER_FEATURES
        ],
        "users": users,
    }


def write_payload(payload: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_js_payload(payload: dict[str, Any], path: str | Path) -> Path:
    """Write the payload as a JS global so the page works from file:// too.

    GitHub Pages serves over http (fetch would work), but embedding the data
    as ``window.BOT_DATA`` means the showcase also opens correctly with a plain
    double-click, with no server and no CORS friction.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, separators=(",", ":"))
    path.write_text(f"window.BOT_DATA = {body};\n", encoding="utf-8")
    return path
