"""
Macroeconomic Risk Scoring Model
=================================
Evaluates country-level risk using weighted economic indicators
and K-Means clustering for segmentation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

DATA = {
    "Country": [
        "Turkey", "Germany", "USA", "Japan", "Brazil",
        "Argentina", "India", "Indonesia", "South Africa", "Mexico",
        "China", "Nigeria", "Egypt", "Saudi Arabia", "Poland"
    ],
    "Inflation":           [64.3,  6.9,  4.1,  3.2, 5.8, 211.4, 5.7,  3.7, 6.9,  4.7,  0.7, 18.8, 33.9,  3.4, 11.4],
    "GDP_Growth":          [ 5.1,  1.9,  2.5,  1.9, 2.9,  -2.5, 7.2,  5.3, 1.9,  3.4,  5.2,  3.3,  3.8,  8.7,  0.9],
    "Unemployment":        [10.1,  3.0,  3.7,  2.6, 8.7,   6.2, 7.5,  5.5, 32.7, 2.8,  5.2,  4.1, 11.9,  4.5,  2.9],
    "Interest_Rate":       [45.0,  4.5,  5.5,  0.1,13.75, 97.0, 6.5,  5.75, 8.25,11.25, 3.45,18.75,21.25, 6.0,  6.75],
    "Debt_to_GDP":         [29.7, 66.3,129.0,255.2,87.9,  88.9,86.9, 37.3, 70.2, 53.8, 51.9, 35.9, 92.7, 24.0, 49.8],
    "Current_Account_Bal": [-4.8,  4.5, -3.0,  3.5,-1.7,   3.5,-1.6, -1.0, -0.3, -1.4,  1.5, -2.4, -3.5,  9.4, -0.7],
    "FX_Reserves_Months":  [ 3.2,  5.2,  4.1, 14.5,16.0,   4.3,11.0,  6.5,  4.2,  4.9, 15.2,  4.7,  3.1, 34.0,  5.3],
    "Budget_Balance":      [-5.2, -2.5, -8.8, -5.6,-6.1,  -6.9,-6.4, -2.7, -6.0, -4.2, -7.8, -6.2,-11.4,  3.0, -5.1],
}

WEIGHTS = {
    "Inflation":           0.20,
    "GDP_Growth":          0.15,
    "Unemployment":        0.10,
    "Interest_Rate":       0.15,
    "Debt_to_GDP":         0.15,
    "Current_Account_Bal": 0.10,
    "FX_Reserves_Months":  0.10,
    "Budget_Balance":      0.05,
}

HIGHER_IS_RISKIER = {
    "Inflation":           True,
    "GDP_Growth":          False,
    "Unemployment":        True,
    "Interest_Rate":       True,
    "Debt_to_GDP":         True,
    "Current_Account_Bal": False,
    "FX_Reserves_Months":  False,
    "Budget_Balance":      False,
}

def build_risk_scores(data, weights, direction):
    df = pd.DataFrame(data)
    scaler = MinMaxScaler()
    indicators = list(weights.keys())

    normalized = pd.DataFrame(
        scaler.fit_transform(df[indicators]),
        columns=indicators
    )

    for col, riskier_if_high in direction.items():
        if not riskier_if_high:
            normalized[col] = 1 - normalized[col]

    w = np.array([weights[c] for c in indicators])
    df["Risk_Score"] = (normalized[indicators].values @ w) * 100

    df["Risk_Category"] = pd.cut(
        df["Risk_Score"],
        bins=[0, 33, 66, 100],
        labels=["Low Risk", "Medium Risk", "High Risk"]
    )

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["Cluster"] = km.fit_predict(normalized[indicators])

    cluster_means = df.groupby("Cluster")["Risk_Score"].mean().sort_values()
    cluster_map = {c: l for c, l in zip(cluster_means.index, ["Low Risk", "Medium Risk", "High Risk"])}
    df["Cluster_Label"] = df["Cluster"].map(cluster_map)

    return df.sort_values("Risk_Score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    result = build_risk_scores(DATA, WEIGHTS, HIGHER_IS_RISKIER)
    print(result[["Country", "Risk_Score", "Risk_Category", "Cluster_Label"]].to_string(index=False))
    result.to_csv("outputs/macro_risk_scores.csv", index=False)
    print("\nSaved to outputs/macro_risk_scores.csv")