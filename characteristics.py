import pandas as pd
import numpy as np

def process_tweets(user_tweets: list, user_metadata: dict) -> dict:
    """
    Calculates characteristics based on a list of post dictionaries 
    and user metadata.
    """
    # Initialize default features
    features = {
        'avg_tweet_len': 0, 
        'avg_hashtags': 0, 
        'avg_links': 0, 
        'avg_mentions': 0,
        'std_tweet_len': 0, 
        'avg_time_gap': 0, 
        'std_time_gap': 0, 
        'min_time_gap': 0,
        # Ensure we keep the original metadata if needed, 
        # though these are likely already in the User object
        'meta_tweet_count': user_metadata.get('tweet_count', 0),
        'meta_z_score': user_metadata.get('z_score', 0)
    }

    if not user_tweets:
        return features

    # Create DataFrame from posts
    df = pd.DataFrame(user_tweets)
    
    # --- Text Analysis ---
    # Ensure text is string to avoid errors
    df['text'] = df['text'].fillna("").astype(str)
    
    df['char_count'] = df['text'].apply(len)
    # simple word count split by whitespace
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['hashtag_count'] = df['text'].apply(lambda x: x.count('#'))
    df['mention_count'] = df['text'].apply(lambda x: x.count('@'))
    df['link_count'] = df['text'].apply(lambda x: x.count('http'))

    features['avg_tweet_len'] = df['char_count'].mean()
    features['std_tweet_len'] = df['char_count'].std()
    features['avg_hashtags'] = df['hashtag_count'].mean()
    features['avg_links'] = df['link_count'].mean()
    features['avg_mentions'] = df['mention_count'].mean()

    # --- Time Analysis ---
    if 'created_at' in df.columns:
        # Convert to datetime, coerce errors to NaT
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        
        # Drop rows where created_at failed to parse
        df = df.dropna(subset=['created_at'])
        
        if not df.empty:
            df = df.sort_values('created_at')
            
            # Calculate time gaps in seconds
            df['time_gap'] = df['created_at'].diff().dt.total_seconds()
            
            # Fill NaNs (the first post has no gap) with 0 or drop it
            time_gaps = df['time_gap'].dropna()
            
            if not time_gaps.empty:
                features['avg_time_gap'] = time_gaps.mean()
                features['std_time_gap'] = time_gaps.std()
                features['min_time_gap'] = time_gaps.min()

    # Replace NaNs with 0 (e.g., if std is NaN because only 1 post exists)
    return {k: (0 if pd.isna(v) else v) for k, v in features.items()}