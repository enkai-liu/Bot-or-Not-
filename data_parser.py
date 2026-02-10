import json
from typing import Dict


class User:
    def __init__(
        self,
        user_id: str,
        username: str = None,
        name: str = None,
        tweet_count: int = 0,
        z_score: float = None,
        description: str = None,
        location: str = None,
    ):
        self.id = user_id
        self.username = username
        self.name = name
        self.tweet_count = tweet_count
        self.z_score = z_score
        self.description = description
        self.location = location
        self.posts = []

    def add_post(self, post: dict):
        self.posts.append(post)

    def __repr__(self):
        return f"<User {self.username} | posts={len(self.posts)}>"

    def __str__(self):
        lines = [
            f"User ID: {self.id}",
            f"Username: {self.username}",
            f"Name: {self.name}",
            f"Tweet Count: {self.tweet_count}",
            f"Z-Score: {self.z_score}",
            f"Location: {self.location}",
            f"Description: {self.description}",
            f"Total Posts: {len(self.posts)}",
        ]

        for i, post in enumerate(self.posts, start=1):
            lines.append(f"  Post {i}:")
            lines.append(f"    ID: {post.get('id')}")
            lines.append(f"    Created At: {post.get('created_at')}")
            lines.append(f"    Language: {post.get('language')}")
            lines.append(f"    Text: {post.get('text')}")

        return "\n".join(lines)


def parse_dataset(raw_json: str) -> Dict:
    data = json.loads(raw_json)

    # --- Metadata ---
    metadata = {
        "id": data.get("id"),
        "language": data.get("lang"),
        "start_time": data["metadata"].get("start_time"),
        "end_time": data["metadata"].get("end_time"),
        "total_users": data["metadata"].get("total_amount_users"),
        "total_posts": data["metadata"].get("total_amount_posts"),
        "avg_posts_per_user": data["metadata"].get("users_average_amount_posts"),
        "avg_z_score": data["metadata"].get("users_average_z_score"),
    }

    # --- Users (id → User object) ---
    users: Dict[str, User] = {}

    for u in data.get("users", []):
        user = User(
            user_id=u.get("id"),
            username=u.get("username"),
            name=u.get("name"),
            tweet_count=u.get("tweet_count"),
            z_score=u.get("z_score"),
            description=u.get("description"),
            location=u.get("location"),
        )
        users[user.id] = user

    # --- Posts (attach to users) ---
    for p in data.get("posts", []):
        post_data = {
            "id": p.get("id"),
            "text": p.get("text"),
            "created_at": p.get("created_at"),
            "language": p.get("lang"),
        }

        author_id = p.get("author_id")

        if author_id in users:
            users[author_id].add_post(post_data)
        else:
            # Handle missing user
            unknown = User(user_id=author_id)
            unknown.add_post(post_data)
            users[author_id] = unknown

    # --- Topics ---
    topics = [
        {"topic": t.get("topic"), "keywords": t.get("keywords", [])}
        for t in data["metadata"].get("topics", [])
    ]

    return {
        "metadata": metadata,
        "topics": topics,
        "users": users,  # User objects
    }


with open("datasets/dataset.posts&users.30.json", "r", encoding="utf-8") as f:
    raw_json = f.read()

parsed = parse_dataset(raw_json)

for user in parsed["users"].values():
    print(user)
    break
