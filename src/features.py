"""Feature engineering.

Each user is reduced to a single numeric vector that describes *how* they post
rather than *what* they post.  The intuition: automated accounts betray
themselves through behavioural fingerprints -- unnaturally regular timing,
atypical text shape, abnormal posting volume -- even when the content itself
looks human.

The public entry point is :func:`extract_features`, which takes the list of a
user's posts plus their account metadata and returns a flat ``dict`` of
features.  Keeping it a plain dict (rather than a hand-maintained schema) means
new signals can be added in one place and automatically flow through the rest
of the pipeline.
"""
from __future__ import annotations

import re
from statistics import mean, pstdev
from typing import Any

from dateutil import parser as date_parser

# Pre-compiled once; `extract_features` is called ~900 times per run.
_URL_RE = re.compile(r"https?://\S+")
_HASHTAG_RE = re.compile(r"#\w+")
_MENTION_RE = re.compile(r"@\w+")
_LETTER_RE = re.compile(r"[A-Za-z]")
_UPPER_RE = re.compile(r"[A-Z]")
_DIGIT_RE = re.compile(r"\d")

# The order here defines the model's feature columns and the order shown in the
# UI, so keep it stable.
FEATURE_NAMES: list[str] = [
    # --- volume ---------------------------------------------------------
    "post_count",
    "reported_tweet_count",
    "posts_per_hour",
    # --- text shape -----------------------------------------------------
    "avg_tweet_len",
    "std_tweet_len",
    "avg_word_count",
    "avg_caps_ratio",
    "avg_digit_ratio",
    "unique_text_ratio",
    # --- entities -------------------------------------------------------
    "avg_hashtags",
    "avg_mentions",
    "url_ratio",
    "hashtag_ratio",
    "mention_ratio",
    # --- timing ---------------------------------------------------------
    "avg_time_gap",
    "std_time_gap",
    "min_time_gap",
    "night_post_ratio",
    # --- account profile ------------------------------------------------
    "z_score",
    "has_description",
    "description_length",
    "has_location",
]

# Human-readable labels + descriptions, consumed by the web showcase so the
# frontend never has to hard-code feature copy.
FEATURE_META: dict[str, dict[str, str]] = {
    "post_count": {"label": "Posts in window", "group": "Volume",
                   "desc": "Number of posts the account made in the sample window."},
    "reported_tweet_count": {"label": "Lifetime tweet count", "group": "Volume",
                             "desc": "Account-level tweet total reported in profile metadata."},
    "posts_per_hour": {"label": "Posts per active hour", "group": "Volume",
                       "desc": "Posting rate across the account's active span."},
    "avg_tweet_len": {"label": "Avg. post length", "group": "Text shape",
                      "desc": "Mean character count per post."},
    "std_tweet_len": {"label": "Post length variability", "group": "Text shape",
                      "desc": "Std-dev of post length; humans vary more than templates."},
    "avg_word_count": {"label": "Avg. word count", "group": "Text shape",
                       "desc": "Mean words per post."},
    "avg_caps_ratio": {"label": "Uppercase ratio", "group": "Text shape",
                       "desc": "Share of letters that are uppercase."},
    "avg_digit_ratio": {"label": "Digit ratio", "group": "Text shape",
                        "desc": "Share of characters that are digits."},
    "unique_text_ratio": {"label": "Unique-post ratio", "group": "Text shape",
                          "desc": "Fraction of posts with distinct text (repetition = automation)."},
    "avg_hashtags": {"label": "Avg. hashtags", "group": "Entities",
                     "desc": "Mean hashtags per post."},
    "avg_mentions": {"label": "Avg. mentions", "group": "Entities",
                     "desc": "Mean @-mentions per post."},
    "url_ratio": {"label": "Link ratio", "group": "Entities",
                  "desc": "Fraction of posts containing a URL."},
    "hashtag_ratio": {"label": "Hashtag ratio", "group": "Entities",
                      "desc": "Fraction of posts containing at least one hashtag."},
    "mention_ratio": {"label": "Mention ratio", "group": "Entities",
                      "desc": "Fraction of posts containing at least one mention."},
    "avg_time_gap": {"label": "Avg. gap between posts", "group": "Timing",
                     "desc": "Mean seconds between consecutive posts."},
    "std_time_gap": {"label": "Posting rhythm variability", "group": "Timing",
                     "desc": "Std-dev of inter-post gaps; bots are often metronomic."},
    "min_time_gap": {"label": "Fastest burst", "group": "Timing",
                     "desc": "Shortest gap between two posts (seconds)."},
    "night_post_ratio": {"label": "Off-hours posting", "group": "Timing",
                         "desc": "Fraction of posts made between 00:00-06:00 UTC."},
    "z_score": {"label": "Activity z-score", "group": "Account profile",
                "desc": "How far the account's activity sits from the population mean."},
    "has_description": {"label": "Has bio", "group": "Account profile",
                        "desc": "Whether the profile has a non-empty description."},
    "description_length": {"label": "Bio length", "group": "Account profile",
                           "desc": "Character length of the profile description."},
    "has_location": {"label": "Has location", "group": "Account profile",
                     "desc": "Whether the profile lists a location."},
}


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _safe_std(values: list[float]) -> float:
    # Population std-dev so a single post yields 0 instead of an error.
    return pstdev(values) if len(values) > 1 else 0.0


def _caps_ratio(text: str) -> float:
    letters = _LETTER_RE.findall(text)
    if not letters:
        return 0.0
    return len(_UPPER_RE.findall(text)) / len(letters)


def _digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(_DIGIT_RE.findall(text)) / len(text)


def extract_features(posts: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, float]:
    """Reduce one user's activity to a numeric feature vector.

    Parameters
    ----------
    posts
        List of post dicts with at least ``text`` and ``created_at`` keys.
    metadata
        Account-level fields: ``tweet_count``, ``z_score``, ``description``,
        ``location``.

    Returns
    -------
    dict
        Feature name -> value, covering every key in :data:`FEATURE_NAMES`.
    """
    texts = [(p.get("text") or "") for p in posts]
    n = len(texts)

    char_lengths = [len(t) for t in texts]
    word_counts = [len(t.split()) for t in texts]
    hashtag_counts = [len(_HASHTAG_RE.findall(t)) for t in texts]
    mention_counts = [len(_MENTION_RE.findall(t)) for t in texts]
    has_url = [1.0 if _URL_RE.search(t) else 0.0 for t in texts]

    # --- timing ---------------------------------------------------------
    timestamps = []
    for p in posts:
        raw = p.get("created_at")
        if not raw:
            continue
        try:
            timestamps.append(date_parser.isoparse(raw))
        except (ValueError, TypeError):
            continue
    timestamps.sort()

    gaps = [
        (timestamps[i] - timestamps[i - 1]).total_seconds()
        for i in range(1, len(timestamps))
    ]
    span_hours = (
        (timestamps[-1] - timestamps[0]).total_seconds() / 3600.0
        if len(timestamps) > 1
        else 0.0
    )
    night_posts = sum(1 for t in timestamps if 0 <= t.hour < 6)

    description = (metadata.get("description") or "").strip()
    location = (metadata.get("location") or "").strip()

    return {
        # volume
        "post_count": float(n),
        "reported_tweet_count": float(metadata.get("tweet_count") or 0),
        "posts_per_hour": (n / span_hours) if span_hours > 0 else 0.0,
        # text shape
        "avg_tweet_len": _safe_mean(char_lengths),
        "std_tweet_len": _safe_std(char_lengths),
        "avg_word_count": _safe_mean(word_counts),
        "avg_caps_ratio": _safe_mean([_caps_ratio(t) for t in texts]),
        "avg_digit_ratio": _safe_mean([_digit_ratio(t) for t in texts]),
        "unique_text_ratio": (len(set(texts)) / n) if n else 0.0,
        # entities
        "avg_hashtags": _safe_mean(hashtag_counts),
        "avg_mentions": _safe_mean(mention_counts),
        "url_ratio": _safe_mean(has_url),
        "hashtag_ratio": _safe_mean([1.0 if c else 0.0 for c in hashtag_counts]),
        "mention_ratio": _safe_mean([1.0 if c else 0.0 for c in mention_counts]),
        # timing
        "avg_time_gap": _safe_mean(gaps),
        "std_time_gap": _safe_std(gaps),
        "min_time_gap": min(gaps) if gaps else 0.0,
        "night_post_ratio": (night_posts / len(timestamps)) if timestamps else 0.0,
        # account profile
        "z_score": float(metadata.get("z_score") or 0.0),
        "has_description": 1.0 if description else 0.0,
        "description_length": float(len(description)),
        "has_location": 1.0 if location else 0.0,
    }
