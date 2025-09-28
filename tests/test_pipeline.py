import pandas as pd
from fraud_risk.data import load_raw
from fraud_risk.model import fit_time_aware

def test_build_and_fit_smoke(tmp_path):
    n = 50
    df = pd.DataFrame({
        "TransactionID": range(n),
        "TransactionDT": range(n),
        "num_a": range(n),
        "cat_b": ["A"] * (n // 2) + ["B"] * (n - n // 2),
        "isFraud": [0] * (n - 5) + [1] * 5,
    })
    raw = tmp_path / "raw"
    raw.mkdir()
    df[["TransactionID", "TransactionDT", "num_a", "cat_b", "isFraud"]].to_csv(raw / "train_transaction.csv", index=False)
    pd.DataFrame({"TransactionID": range(n)}).to_csv(raw / "train_identity.csv", index=False)

    X, y = load_raw(raw)
    pipe, res = fit_time_aware(X, y, n_splits=3)

    assert 0.0 <= res.auc_mean <= 1.0
    assert 0.0 <= res.ap_mean <= 1.0
