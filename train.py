"""Reproducible baseline entry point for the supplied used-car dataset.

Replaces the broken fit_handle_* API sequence with train-only sklearn pipelines.
The exploratory notebooks and original handling_data class remain for study.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def run():
    data = pd.read_csv(Path(__file__).with_name('car_price.csv')).drop_duplicates()
    data['price'] = pd.to_numeric(data['price'], errors='coerce')
    data = data.dropna(subset=['price'])
    data = data.loc[data.price > 0].copy()
    X = data.drop(columns='price')
    y = data.price
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=.2, random_state=42)
    nums = X.select_dtypes(include='number').columns.tolist()
    cats = X.select_dtypes(exclude='number').columns.tolist()
    def prep():
        return ColumnTransformer([
            ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), nums),
            ('category', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore'))]), cats)
        ])
    regression = Pipeline([('prepare', prep()), ('model', Ridge(alpha=10))])
    regression.fit(train_X, train_y)
    prediction = regression.predict(test_X)
    # Category boundaries are learned only from training prices.
    low, high = train_y.quantile([1/3, 2/3]).values
    def tier(values): return np.where(values <= low, 'cheap', np.where(values <= high, 'moderate', 'expensive'))
    classification = Pipeline([('prepare', prep()), ('model', KNeighborsClassifier(n_neighbors=7, n_jobs=2))])
    classification.fit(train_X, tier(train_y))
    predicted_tier = np.concatenate([classification.predict(test_X.iloc[start:start+128]) for start in range(0,len(test_X),128)])
    metrics = {'rows': len(data), 'split': 'random 80/20, seed 42', 'regression_model': 'Ridge alpha=10', 'mae': mean_absolute_error(test_y,prediction), 'rmse': root_mean_squared_error(test_y,prediction), 'r2': r2_score(test_y,prediction), 'knn_tier_accuracy': accuracy_score(tier(test_y),predicted_tier), 'train_only_price_cutoffs': [float(low),float(high)]}
    print(json.dumps(metrics,indent=2))
    return metrics

if __name__ == '__main__':
    run()
