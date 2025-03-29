"""
Creates features that describe how close defenders are to the quarterback over time.
"""

import pandas as pd
import numpy as np

def get_qb_distance_features(tracking_df, players_df):
    """
    Calculates distance from each player to the QB at each frame of a play

    Parameters:
        - tracking_df (pd.DataFrame): Tracking data from weekX.csv
        - players_df (pd.DataFrame): Players metadata, includes officialPosition

    Returns:
        - pd.DataFrame: Original tracking data with added distanceToQB column
    """
    players_df = players_df.rename(columns={'officialPosition': 'position'})
    
    # merge positions into tracking data
    tracking_df = tracking_df.merge(players_df[['nflId', 'position']], on='nflId', how='left')

    features = []

    # Process each play
    for (gameId, playId), group in tracking_df.groupby(['gameId', 'playId']):
        # Find QB on the play
        qb_rows = group[group['position'] == 'QB']
        if qb_rows.empty:
            continue

        # only one QB per play
        qb_id = qb_rows['nflId'].unique()[0]

        # Process each frame in the play
        # Group by frameId to process each frame separately so we can calculate distance for each frame
        for frameId, frame_data in group.groupby('frameId'):
            qb_data = frame_data[frame_data['nflId'] == qb_id]
            if qb_data.empty:
                continue

            qb_x = qb_data.iloc[0]['x']
            qb_y = qb_data.iloc[0]['y']

            # Add distanceToQB for all players in this frame
            frame_data = frame_data.copy()
            frame_data['distanceToQB'] = np.sqrt((frame_data['x'] - qb_x)**2 + (frame_data['y'] - qb_y)**2)
            frame_data['qbId'] = qb_id
            features.append(frame_data)

    return pd.concat(features, ignore_index=True)
