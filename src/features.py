"""Ingeniería de features para Ames Housing (misma lógica que el notebook EC2).

Transformador sin estado (no aprende nada de los datos), por lo que no introduce leakage.
"""
from sklearn.base import BaseEstimator, TransformerMixin

QUAL_MAP = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ORDINAL_COLS = ["Exter Qual", "Exter Cond", "Bsmt Qual", "Bsmt Cond", "Heating QC",
                "Kitchen Qual", "Fireplace Qu", "Garage Qual", "Garage Cond", "Pool QC"]


class AmesFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in ORDINAL_COLS:
            X[c] = X[c].map(QUAL_MAP).fillna(0)
        X["TotalSF"] = X[["Total Bsmt SF", "1st Flr SF", "2nd Flr SF"]].fillna(0).sum(axis=1)
        X["TotalBath"] = (X["Full Bath"].fillna(0) + 0.5 * X["Half Bath"].fillna(0)
                          + X["Bsmt Full Bath"].fillna(0) + 0.5 * X["Bsmt Half Bath"].fillna(0))
        X["HouseAge"] = X["Yr Sold"] - X["Year Built"]
        X["YearsSinceRemod"] = X["Yr Sold"] - X["Year Remod/Add"]
        X["PorchSF"] = X[["Open Porch SF", "Enclosed Porch", "3Ssn Porch", "Screen Porch", "Wood Deck SF"]].sum(axis=1)
        X["QualxArea"] = X["Overall Qual"] * X["Gr Liv Area"]
        X["HasGarage"] = X["Garage Area"].fillna(0).gt(0).astype(int)
        X["HasBsmt"] = X["Total Bsmt SF"].fillna(0).gt(0).astype(int)
        return X
