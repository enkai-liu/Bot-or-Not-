"""Load the raw competition data into tidy Python objects.

The dataset ships as, per round:

* ``dataset.posts&users.NN.json`` -- users + posts for round ``NN``
* ``dataset.bots.NN.txt``         -- newline-separated UUIDs of the bots

This module joins posts to their authors and attaches the ground-truth label,
exposing a single :func:`load_users` helper that the rest of the pipeline
builds on.  Bot UUIDs live *only* in the ``.txt`` files -- there is deliberately
no hard-coded list anywhere in the code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"


@dataclass
class User:
    """One account: profile metadata, its posts, and the ground-truth label."""

    user_id: str
    dataset: str
    username: str | None = None
    name: str | None = None
    tweet_count: int = 0
    z_score: float | None = None
    description: str | None = None
    location: str | None = None
    posts: list[dict[str, Any]] = field(default_factory=list)
    is_bot: bool = False

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "tweet_count": self.tweet_count,
            "z_score": self.z_score,
            "description": self.description,
            "location": self.location,
        }


def discover_rounds(datasets_dir: Path = DATASETS_DIR) -> list[str]:
    """Return the round ids (e.g. ``["30", "31", ...]``) present on disk."""
    ids = []
    for path in sorted(datasets_dir.glob("dataset.posts&users.*.json")):
        # Filename pattern: dataset.posts&users.<id>.json
        ids.append(path.name.split(".")[-2])
    return ids


def _load_bot_ids(round_id: str, datasets_dir: Path) -> set[str]:
    bot_file = datasets_dir / f"dataset.bots.{round_id}.txt"
    if not bot_file.exists():
        return set()
    return {
        line.strip()
        for line in bot_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def load_round(round_id: str, datasets_dir: Path = DATASETS_DIR) -> list[User]:
    """Load every user (with posts + label) for a single round."""
    json_path = datasets_dir / f"dataset.posts&users.{round_id}.json"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    bot_ids = _load_bot_ids(round_id, datasets_dir)

    users: dict[str, User] = {}
    for u in data.get("users", []):
        uid = u.get("id")
        users[uid] = User(
            user_id=uid,
            dataset=round_id,
            username=u.get("username"),
            name=u.get("name"),
            tweet_count=u.get("tweet_count") or 0,
            z_score=u.get("z_score"),
            description=u.get("description"),
            location=u.get("location"),
            is_bot=uid in bot_ids,
        )

    # Attach posts; create a stub user if a post references an unknown author.
    for p in data.get("posts", []):
        author_id = p.get("author_id")
        if author_id not in users:
            users[author_id] = User(
                user_id=author_id, dataset=round_id, is_bot=author_id in bot_ids
            )
        users[author_id].posts.append(
            {
                "id": p.get("id"),
                "text": p.get("text"),
                "created_at": p.get("created_at"),
                "lang": p.get("lang"),
            }
        )

    return list(users.values())


def load_users(
    round_ids: list[str] | None = None, datasets_dir: Path = DATASETS_DIR
) -> list[User]:
    """Load and concatenate users across the requested rounds (all by default)."""
    if round_ids is None:
        round_ids = discover_rounds(datasets_dir)
    users: list[User] = []
    for rid in round_ids:
        users.extend(load_round(rid, datasets_dir))
    return users
