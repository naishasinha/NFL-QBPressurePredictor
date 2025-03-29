"""
TEST FEATURES OUTPUT

This script tests the feature generation for distance to QB from defenders.
"""

import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_tracking_week, load_players_data
from src.feature_utils import get_qb_distance_features

# Load data
df = load_tracking_week(1)
players_df = load_players_data()

# Pick one play to test
sample_play = df[(df['gameId'] == df['gameId'].iloc[0]) & (df['playId'] == df['playId'].iloc[0])]

# Generate features
features = get_qb_distance_features(sample_play, players_df)

print("\n=== Features Output ===")
print(features[['gameId', 'playId', 'frameId', 'nflId', 'position', 'x', 'y', 'distanceToQB']].head())