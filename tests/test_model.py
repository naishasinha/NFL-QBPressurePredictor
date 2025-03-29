"""
TEST MODEL OUTPUT

This script tests the full pipeline: feature engineering, labeling, and model training.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_tracking_week, load_players_data
from src.feature_utils import get_qb_distance_features
from src.labeling import label_pressure_events
from src.model import extract_play_level_features, train_and_evaluate_model

# Load data
print("Loading week 1 tracking data...")
tracking_df = load_tracking_week(1)
players_df = load_players_data()

# Generate features for the entire dataset (feature engineering)
print("Generating distance features...")
features_df = get_qb_distance_features(tracking_df, players_df)

# Label the plays based on pressure events (labeling)
print("Labeling pressure events...")
labeled_df = label_pressure_events(features_df)

# Extract play-level features for modeling
print("Extracting play-level features...")
play_level_features_df = extract_play_level_features(features_df)

# Merge features with labels
print("Merging features with labels...")
merged = play_level_features_df.merge(
    labeled_df[['gameId', 'playId', 'pressure']], 
    on=['gameId', 'playId'], 
    how='inner'
)

# Prepare the feature set (X) and target variable (y)
print("Preparing feature set and target variable...")
X = merged.drop(columns=['gameId', 'playId', 'pressure'])  # Features for modeling
y = merged['pressure']  # Target variable (0 or 1 for pressure)

# Train and evaluate models
print("Training and evaluating models...")
train_and_evaluate_model(X, y)