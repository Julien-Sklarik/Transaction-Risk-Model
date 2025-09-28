from pathlib import Path
import json
import pandas as pd

from fraud_risk.data import load_raw
from fraud_risk.model import fit_time_aware, save_model
from fraud_risk.evaluate import compute_curves, choose_threshold_from_pr
from fraud_risk.plotting import save_pr_curve, save_roc_curve

def main():
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    model_dir = root / "models"
    reports_dir = root / "reports"
    oof_path = reports_dir / "tables" / "oof_predictions.csv"
    metrics_path = reports_dir / "tables" / "cv_metrics.json"

    X, y = load_raw(raw_dir)
    pipe, cvres = fit_time_aware(X, y)

    save_model(pipe, model_dir / "fraud_model.joblib")

    oof_df = pd.DataFrame({"oof_proba": cvres.oof_proba, "y": y.values})
    oof_df.to_csv(oof_path, index=False)

    curves = compute_curves(y.values, cvres.oof_proba)
    save_pr_curve(curves["precision"], curves["recall"], reports_dir / "figures" / "pr_curve.png")
    save_roc_curve(curves["fpr"], curves["tpr"], reports_dir / "figures" / "roc_curve.png")

    thr = choose_threshold_from_pr(y.values, cvres.oof_proba)
    metrics = {
        "auc_mean": cvres.auc_mean,
        "auc_std": cvres.auc_std,
        "ap_mean": cvres.ap_mean,
        "ap_std": cvres.ap_std,
        "chosen_threshold": thr.threshold,
        "f1_at_threshold": thr.f1,
        "precision_at_threshold": thr.precision,
        "recall_at_threshold": thr.recall
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
