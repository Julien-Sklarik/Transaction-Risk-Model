from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt

def save_pr_curve(precision, recall, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.step(recall, precision, where="post")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

def save_roc_curve(fpr, tpr, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
