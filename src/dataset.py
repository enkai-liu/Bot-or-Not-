"""Assemble the user-level feature table consumed by the model."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import parser
from .features import FEATURE_NAMES, extract_features


def build_feature_table(round_ids: list[str] | None = None) -> pd.DataFrame:
    """Return a tidy DataFrame: one row per user, feature columns + label.

    Columns: ``user_id``, ``username``, ``dataset``, every name in
    :data:`features.FEATURE_NAMES`, and the integer ``is_bot`` target.
    """
    users = parser.load_users(round_ids)
    records = []
    for user in users:
        feats = extract_features(user.posts, user.metadata)
        records.append(
            {
                "user_id": user.user_id,
                "username": user.username or "unknown",
                "dataset": user.dataset,
                **feats,
                "is_bot": int(user.is_bot),
            }
        )

    df = pd.DataFrame(records)
    # Guarantee column order: identifiers, features, then label.
    ordered = ["user_id", "username", "dataset", *FEATURE_NAMES, "is_bot"]
    return df[ordered]


def save_feature_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    table = build_feature_table()
    print(f"Built feature table: {table.shape[0]} users x {table.shape[1]} columns")
    print(f"Bots: {int(table['is_bot'].sum())} / {len(table)}")
    save_feature_table(table, "data/features.csv")
    print("Wrote data/features.csv")
