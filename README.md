<div align="center">
  
# NFL Quarterback Pressure Predictor: ML-Based Analysis of Defensive Engagement

***
![giphy](https://github.com/user-attachments/assets/6106f50a-7eba-4f9a-b369-1534a6b2abf1)
***
</div>


## Objective
To develop a real-time predictive Machine Learning model and dashboard that can:
- Analyze tracking data on a 0.1-second level
- Estimate the probability of QB pressure based on player movement
- Visualize pressure likelihood during live plays

NFL coaches care about pressure situations because they directly impact QB performance. This tool (**_made for demostration and learning purposes only_**) was developed to quantify and explore how likely a QB is to be pressured in real-time can help evaluate O-line performance, design plays that minimize pressure risk, and assess defensive effectiveness.

## Repository Structure
```
qb-pressure-tracker/
│
├── data/                     # Raw and processed datasets
│   ├── raw/                  # Original NFL datasets
│   └── processed/            # Cleaned and feature-engineered data
│
├── notebooks/               # Jupyter notebooks for exploration and modeling
│   ├── 01_data_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── src/                     # Source code
│   ├── data_loader.py       # Data loading functions
│   ├── feature_utils.py     # Feature engineering helpers
│   ├── model.py             # Model training and inference
│   └── visualization.py     # Visualization utilities
│
├── app/                     # Dashboard or deployment code (Streamlit, etc.)
│   └── dashboard.py
│
├── tests/                   # Unit tests for code modules
│   ├── test_features.py     # Tests feature engineering (distance to QB)
│   ├── test_labeling.py     # Tests labeling logic (pressure events)
│   └── test_model.py        # Test model training and evaluation
│
├── README.md                # Project overview and setup
├── requirements.txt         # Python dependencies
└── .gitignore
```

## Sample Features Used
- Distance and speed between QB and nearest defenders
- Directional angles of approach
- QB time-to-throw and initial positioning
- Play context (down, distance, formation, etc.)

## Workflow
1. **Data Cleaning** – Load and preprocess tracking + play-by-play data
2. **Feature Engineering** – Extract time-series features on 0.1s intervals
3. **Modeling** – Train a classifier (e.g., XGBoost) to predict pressure events
4. **Deployment** – Display results in a simple web interface (e.g., Streamlit)

## Installation
```bash
git clone https://github.com/yourusername/qb-pressure-tracker.git
cd qb-pressure-tracker
pip install -r requirements.txt
```

## Running the Dashboard
```bash
streamlit run app/dashboard.py
```

## References
- [NFL Big Data Bowl Dataset](https://www.kaggle.com/competitions/nfl-big-data-bowl-2023/data)
- [NFLFastR](https://www.nflfastr.com/)
