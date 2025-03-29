"""
Labels estimated pressure events for the quarterback

Pressure Heuristic:
- If any defender gets within 1.5 yards of the QB within the first 2.5 seconds of the play (25 frames at 10 fps)
"""

import pandas as pd

def label_pressure_events(tracking_df, threshold_yards=1.5, window_frames=25):
    """
    Labels each (gameId, playId) as a pressure play if any defender is
    within threshold_yards of the quarterback within a specified window of frames (2.5 seconds).

    Parameters:
        - tracking_df (pd.DataFrame): Tracking data with distanceToQB and position columns
        - threshold_yards (float): Distance in yards to consider as pressure (default is 1.5 yards)
        - window_frames (int): Number of frames to check for pressure (default is 25 frames, which is 2.5 seconds at 10 fps)

    Returns:
        - pd.DataFrame: Labals per (gameId, playId) with pressure = 1 or 0
    """
    pressure_labels = []

    # Group by play
    for (gameId, playId), group in tracking_df.groupby(['gameId', 'playId']):
        # only look at first 2.5 seconds of frames
        group = group[group['frameId'] <= group['frameId'].min() + window_frames]

        # Filter to only defenders (no QB or QB's team)
        qb_team = group[group['nflId'] == group['qbId'].iloc[0]]['team'].iloc[0]
        defenders = group[group['team'] != qb_team]

        # Check if any defender is within  threshold distance to the QB
        pressure = int((defenders['distanceToQB'] <= threshold_yards).any())

        pressure_labels.append({
            'gameId': gameId,
            'playId': playId,
            'pressure': pressure
        })

    return pd.DataFrame(pressure_labels)