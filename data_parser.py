import json
import pandas as pd
from typing import Dict, List
from pathlib import Path
from datetime import datetime

from characteristics import process_tweets

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
        self.characteristics = {}

    def add_post(self, post: dict):
        self.posts.append(post)

    def calculate_stats(self):
        """
        Passes full post objects and user metadata to process_tweets
        to generate the required columns.
        """
        user_meta = {
            "tweet_count": self.tweet_count,
            "z_score": self.z_score,
            "description": self.description
        }
        
        if self.posts:
            self.characteristics = process_tweets(self.posts, user_meta)
        else:
            self.characteristics = process_tweets([], user_meta)

    def to_rows(self):
        if not self.posts:
            base_data = {
                "user_id": self.id,
                "username": self.username,
                "name": self.name,
                "location": self.location,
                "z_score": self.z_score,
                "post_id": None,
                "text": None,
                "created_at": None,
                "lang": None
            }
            base_data.update(self.characteristics)
            return [base_data]
        
        rows = []
        for p in self.posts:
            row_data = {
                "user_id": self.id,
                "username": self.username,
                "name": self.name,
                "location": self.location,
                "z_score": self.z_score,
                "post_id": p.get("id"),
                "text": p.get("text"),
                "created_at": p.get("created_at"),
                "lang": p.get("language")
            }
            row_data.update(self.characteristics)
            rows.append(row_data)
        return rows

    def __repr__(self):
        return f"<User {self.username} | posts={len(self.posts)}>"

def parse_dataset(raw_json: str) -> Dict:
    data = json.loads(raw_json)

    metadata = {
        "id": data.get("id"),
        "total_users": data["metadata"].get("total_amount_users"),
        "total_posts": data["metadata"].get("total_amount_posts"),
    }

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

def main():
    folder_name = "datasets"
    
    print(f"Scanning directory: ./{folder_name}/")
    file_input = input("Enter JSON filenames separated by commas (e.g., file1.json, file2.json): ")
    
    filenames = [name.strip() for name in file_input.split(",") if name.strip()]
    
    if not filenames:
        print("No filenames provided. Exiting.")
        return

    all_rows = []

    for filename in filenames:
        file_path = Path(folder_name) / filename
        
        print(f"--- Processing {filename} ---")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_json = f.read()

            parsed = parse_dataset(raw_json)
            
            file_user_count = 0
            for user in parsed["users"].values():
                user.calculate_stats()
                user_rows = user.to_rows()
                all_rows.extend(user_rows)
                file_user_count += 1
            
            print(f"    Successfully processed {file_user_count} users from {filename}.")

        except FileNotFoundError:
            print(f"    Error: The file '{filename}' was not found in '{folder_name}'. Skipping.")
        except json.JSONDecodeError:
            print(f"    Error: Failed to decode JSON from '{filename}'. Skipping.")
        except Exception as e:
            print(f"    An unexpected error occurred processing '{filename}': {e}")

    if all_rows:
        df = pd.DataFrame(all_rows)

        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])

        print("\n--- Aggregation Complete ---")
        print(f"Total Combined Shape: {df.shape}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f'parsed_data_combined_{timestamp}.csv'
        
        try:
            df.to_csv(output_filename, index=False)
            print(f"Successfully saved {len(df)} rows to {output_filename}")
        except PermissionError:
            print(f"Error: Could not save to {output_filename}. Is the file open?")
    else:
        print("\nNo data was processed successfully.")

if __name__ == "__main__":
    main()