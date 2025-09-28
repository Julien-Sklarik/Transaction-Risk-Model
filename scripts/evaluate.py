from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import classification_report

def main():
    root = Path(__file__).resolve().parents[1]
    model_path = root / "models" / "fraud_model.joblib"
    holdout_path = root / "data" / "processed" / "holdout.csv"

    model = joblib.load(model_path)
    df = pd.read_csv(holdout_path)
    y = df.pop("isFraud").astype(int)
    X = df
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    rep = classification_report(y, pred, digits=3)
    print(rep)

if __name__ == "__main__":
    main()
