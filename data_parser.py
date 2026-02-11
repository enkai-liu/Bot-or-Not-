import json
import pandas as pd
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

    def to_rows(self):
        """Flattens user and post data into a list of dictionaries for Pandas."""
        if not self.posts:
            # Return one row with empty post data if user has no posts
            return [{**self.__dict__, "post_id": None, "text": None, "post_at": None}]
        
        rows = []
        for p in self.posts:
            rows.append({
                "user_id": self.id,
                "username": self.username,
                "name": self.name,
                "location": self.location,
                "z_score": self.z_score,
                "post_id": p.get("id"),
                "text": p.get("text"),
                "created_at": p.get("created_at"),
                "lang": p.get("language")
            })
        return rows

    def __repr__(self):
        return f"<User {self.username} | posts={len(self.posts)}>"

def parse_dataset(raw_json: str) -> Dict:
    data = json.loads(raw_json)

    # --- Metadata & Topics (Same as your original) ---
    metadata = {
        "id": data.get("id"),
        "total_users": data["metadata"].get("total_amount_users"),
        "total_posts": data["metadata"].get("total_amount_posts"),
    }

    # --- Users & Posts ---
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

    for p in data.get("posts", []):
        post_data = {
            "id": p.get("id"),
            "text": p.get("text"),
            "created_at": p.get("created_at"),
            "language": p.get("lang"),
        }
        author_id = p.get("author_id")
        if author_id not in users:
            users[author_id] = User(user_id=author_id)
        users[author_id].add_post(post_data)

    return {"metadata": metadata, "users": users}

# --- Main Execution ---
with open("dataset.posts&users.30.json", "r", encoding="utf-8") as f:
    raw_json = f.read()

parsed = parse_dataset(raw_json)

# CONVERSION TO DATAFRAME
# We flatten the list of lists created by to_rows()
all_rows = [row for user in parsed["users"].values() for row in user.to_rows()]
df = pd.DataFrame(all_rows)

# Optional: Clean up the date column
df['created_at'] = pd.to_datetime(df['created_at'])

print(df.head())

parsed = parse_dataset(raw_json)

# 1. Convert your objects to a list of dictionaries
all_rows = [user.to_rows() for user in parsed["users"].values()]

# 2. Create the DataFrame
df_to_save = pd.DataFrame(all_rows)

# 3. SAVE IT TO DISK (This creates the file your ML script is looking for)
df_to_save.to_csv('parsed_data.csv', index=False)