"""Sanity tests for parsing and feature extraction.

Run with:  python -m pytest    (or just: python tests/test_pipeline.py)
These guard the contract the model and frontend depend on -- stable feature
names, sane ranges, and labels loaded from the bot files rather than guessed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import FEATURE_NAMES, extract_features  # noqa: E402
from src.parser import discover_rounds, load_round  # noqa: E402


def test_feature_vector_matches_schema():
    posts = [
        {"text": "Check this out https://t.co/x #news @alice",
         "created_at": "2024-03-16T00:00:00Z"},
        {"text": "Another post about #news today",
         "created_at": "2024-03-16T01:00:00Z"},
    ]
    meta = {"tweet_count": 100, "z_score": 1.5,
            "description": "hi", "location": "NYC"}
    feats = extract_features(posts, meta)

    # Every declared feature is present, and nothing extra leaks in.
    assert set(feats) == set(FEATURE_NAMES)
    # All values are finite numbers.
    assert all(isinstance(v, (int, float)) for v in feats.values())


def test_feature_values_are_sensible():
    posts = [
        {"text": "HELLO WORLD 123", "created_at": "2024-03-16T00:00:00Z"},
        {"text": "HELLO WORLD 123", "created_at": "2024-03-16T00:00:30Z"},
    ]
    feats = extract_features(posts, {"tweet_count": 2, "z_score": 0})
    assert feats["post_count"] == 2
    assert feats["unique_text_ratio"] == 0.5  # one duplicate of two
    assert feats["min_time_gap"] == 30.0
    assert 0.0 <= feats["avg_caps_ratio"] <= 1.0


def test_empty_user_is_safe():
    feats = extract_features([], {"tweet_count": 0, "z_score": None})
    assert feats["post_count"] == 0
    assert all(v == 0 for v in feats.values())


def test_labels_come_from_bot_files():
    rounds = discover_rounds()
    assert rounds, "no datasets discovered"
    users = load_round(rounds[0])
    assert users
    # Round 30 is known to contain labelled bots.
    assert any(u.is_bot for u in users)
    assert any(not u.is_bot for u in users)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("All tests passed.")
