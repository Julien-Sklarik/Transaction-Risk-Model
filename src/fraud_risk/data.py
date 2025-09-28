from pathlib import Path
import pandas as pd

def load_raw(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    raw_dir = Path(raw_dir)
    tx = pd.read_csv(raw_dir / "train_transaction.csv")
    idn = pd.read_csv(raw_dir / "train_identity.csv")
    df = tx.merge(idn, on="TransactionID", how="left").sort_values("TransactionDT")
    y = df.pop("isFraud").astype(int)
    X = df.drop(columns=["TransactionID"], errors="ignore")
    return X, y
