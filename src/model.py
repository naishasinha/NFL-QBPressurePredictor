"""
Train and evaluate using two models (LogisticRegression and XGBoostClassifier)
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report  
from xgboost import XGBClassifier

def extract_play_level_features(tracking_df):
    """
    Collapse frame-level data into play-level features for modeling.

    For each (gameId, playId), we'll create:
    - minDistanceToQB: Minimum distance of any defender to the QB
    - meanDistanceToQB: Mean distance of all defenders to the QB
    - stdDistanceToQB: Standard deviation of distances of defenders to the QB
    - maxSpeedNearQB: (defender speed near QB)
    """
    play_features = []

    for (gameId, playId), group in tracking_df.groupby(['gameId', 'playId']):
        # only look at defenders
        try:
            qb_id = group['qbId'].iloc[0]
            qb_team = group[group['nflId'] == qb_id]['team'].iloc[0]
        except (IndexError, KeyError):
            continue  # skip if missing
        
        defenders = group[group['team'] != qb_team]

        # only close defenders to the QB within 5 yards
        close_defenders = defenders[defenders['distanceToQB'] <= 5]
        if close_defenders.empty:
            continue

        features = {
            'gameId': gameId,
            'playId': playId,
            'minDistanceToQB': close_defenders['distanceToQB'].min(),
            'meanDistanceToQB': close_defenders['distanceToQB'].mean(),
            'stdDistanceToQB': close_defenders['distanceToQB'].std(),
        }

        play_features.append(features)
    
    return pd.DataFrame(play_features)


def train_and_evaluate_model(X, y):
    """
    Train and evaluate two models: Logistic Regression and XGBoost Classifier.

    Parameters:
    - X: Feature DataFrame
    - y: Labels (target variable)

    Returns:
    - None
    """
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression Model
    print("=== Logistic Regression Model ===")
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred_logreg = logreg.predict(X_test)
    print(classification_report(y_test, y_pred_logreg, target_names=['No Pressure', 'Pressure']))

    # XGBoost Classifier Model
    print("=== XGBoost Classifier Model ===")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    print(classification_report(y_test, y_pred_xgb, target_names=['No Pressure', 'Pressure']))

