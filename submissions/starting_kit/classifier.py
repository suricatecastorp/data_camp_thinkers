from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
import numpy as np


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = RandomForestRegressor(n_estimators=5)

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        y_pred = np.array([int(x > 0.5) for x in self.reg.predict(X)])
        return np.array([1 - y_pred, y_pred]).T
