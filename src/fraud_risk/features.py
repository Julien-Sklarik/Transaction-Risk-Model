from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    num = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(with_mean=True)
    )

    cat = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num, num_cols),
            ("cat", cat, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return pre
