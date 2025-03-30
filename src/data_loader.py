"""
Data Cleaning for the Tracking and Play-By-Play Data
"""

import pandas as pd
import os

def load_tracking_week(number: int, base_path="data/raw"):
    """
    Load NFL tracking data for a specific week
    """
    file_path = os.path.join(base_path, f"week{number}.csv")
    return pd.read_csv(file_path)

def load_all_tracking_weeks(base_path="data/raw", weeks=range(1, 4)):
    """
    Load all weekly tracking data into a dictionary of DataFrames
    """
    dfs = [load_tracking_week(week, base_path) for week in weeks]
    return pd.concat(dfs, ignore_index=True)

def concatenate_tracking_weeks(base_path="data/raw", weeks=range(1, 9)):
    """
    Load all weekly tracking data and concatenate into a single DF
    """
    dfs = [load_tracking_week(week, base_path) for week in weeks]
    return pd.concat(dfs, ignore_index=True)

def load_plays_data(base_path="data/raw"):
    """
    Load the metadata of the plays
    """
    file_path = os.path.join(base_path, "plays.csv")
    return pd.read_csv(file_path)

def load_players_data(base_path="data/raw"):
    """
    Load the metadata of the players
    """
    file_path = os.path.join(base_path, "players.csv")
    return pd.read_csv(file_path)

def load_games_data(base_path="data/raw"):
    """
    Load the metadata of the games
    """
    file_path = os.path.join(base_path, "games.csv")
    return pd.read_csv(file_path)