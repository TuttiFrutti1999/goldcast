"""Prognosemodelle. Alle sagen die Log-Rendite über h Handelstage voraus."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class Drift:
    name = "Historischer Drift"

    def __init__(self, seed=42):
        self.seed = seed

    def fit(self, X, y, **_):
        self.mu = float(y.iloc[-2520:].mean()) if len(y) else 0.0
        return self

    def predict(self, X):
        return np.full(len(X), self.mu)


class BayesRidge:
    """Bayesianische lineare Regression (multiple Regression mit Prior)."""
    name = "Bayes-Regression"

    def __init__(self, seed=42):
        self.seed = seed

    def fit(self, X, y, **_):
        from sklearn.linear_model import BayesianRidge

        self.med = X.median()
        Xf = X.fillna(self.med)
        self.mu, self.sd = Xf.mean(), Xf.std().replace(0, 1)
        self.m = BayesianRidge().fit(((Xf - self.mu) / self.sd).values, y.values)
        return self

    def predict(self, X):
        Xf = X.fillna(self.med)
        return self.m.predict(((Xf - self.mu) / self.sd).values)


class GBM:
    name = "LightGBM"

    def __init__(self, seed=42):
        self.seed = seed

    def fit(self, X, y, **_):
        import lightgbm as lgb

        self.m = lgb.LGBMRegressor(n_estimators=250, learning_rate=0.03, num_leaves=15, max_depth=4,
                                   min_child_samples=80, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.7, reg_lambda=5.0, random_state=self.seed, verbose=-1)
        self.m.fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X)

    def contributions(self, X):
        """SHAP-Beiträge (TreeSHAP, in LightGBM eingebaut)."""
        return self.m.predict(X, pred_contrib=True)


class GBMDirection:
    """Klassifikator für P(Anstieg)."""
    name = "LightGBM-Richtung"

    def __init__(self, seed=42):
        self.seed = seed

    def fit(self, X, y, **_):
        import lightgbm as lgb

        self.m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=15, max_depth=4,
                                    min_child_samples=80, subsample=0.8, subsample_freq=1,
                                    colsample_bytree=0.7, reg_lambda=5.0, random_state=self.seed, verbose=-1)
        self.m.fit(X, (y > 0).astype(int))
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)[:, 1]


class RandomForest:
    name = "Random Forest"

    def __init__(self, seed=42):
        self.seed = seed

    def fit(self, X, y, **_):
        from sklearn.ensemble import RandomForestRegressor

        self.med = X.median()
        self.m = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=60, max_features=0.4, max_samples=0.5,
                                       n_jobs=-1, random_state=self.seed)
        self.m.fit(X.fillna(self.med).values, y.values)
        return self

    def predict(self, X):
        return self.m.predict(X.fillna(self.med).values)


# ------------------------------------------------------------ Zeitreihenmodelle (auf Tagesrenditen)
class ARIMA:
    name = "ARIMA"

    def fit_series(self, r: pd.Series):
        from statsmodels.tsa.arima.model import ARIMA as SM

        r = r.dropna().iloc[-1500:]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.res = SM(r.values, order=(1, 0, 1)).fit()
        except Exception:  # noqa: BLE001
            self.res = None
        self.mu = float(r.mean())
        return self

    def forecast_cum(self, h: int) -> float:
        if self.res is None:
            return self.mu * h
        return float(np.sum(self.res.forecast(h)))


class GARCH:
    """GARCH(1,1) mit t-Verteilung – Volatilitätsprognose."""
    name = "GARCH"

    def fit_series(self, r: pd.Series):
        from arch import arch_model

        r = r.dropna().iloc[-2500:] * 100
        self.fallback = float(r.iloc[-63:].std())
        try:
            self.res = arch_model(r.values, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
        except Exception:  # noqa: BLE001
            self.res = None
        return self

    def sigma_h(self, h: int) -> float:
        """Std.-Abw. der kumulierten h-Tages-Log-Rendite (Dezimal)."""
        if self.res is None:
            return self.fallback / 100 * np.sqrt(h)
        var = self.res.forecast(horizon=h, reindex=False).variance.values[-1]
        return float(np.sqrt(var.sum())) / 100

    def daily_vol_annual(self) -> float:
        return self.sigma_h(1) * np.sqrt(252) * 100


FEATURE_MODELS = [Drift, BayesRidge, GBM, RandomForest]
