"""
TEST LABELING OUTPUT

This script tests the labeling function for plays to ensure it correctly labels the plays based on
how close defenders are to the quarterback within the first 2.5 seconds of the play.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_tracking_week, load_players_data
from src.feature_utils import get_qb_distance_features
from src.labeling import label_pressure_events

# Load data
df = load_tracking_week(1)
players_df = load_players_data()

# Pick one play to test
sample_play = df[(df['gameId'] == df['gameId'].iloc[0]) & (df['playId'] == df['playId'].iloc[0])]

# Generate features
features = get_qb_distance_features(sample_play, players_df)
labels_df = label_pressure_events(features)

print("\n=== Pressure Labels ===")
print(labels_df)
