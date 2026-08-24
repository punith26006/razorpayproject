"""Cost-Curve Evaluation Module for Return-Risk Scorer.

Optimizes decision thresholds based on real asymmetric business costs (₹)
rather than standard F1 maximization.
"""

import json
import logging
import os
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


class CostCurveEvaluator:
    """Evaluates classifier performance across operating thresholds using financial loss curves."""

    def __init__(self, cost_fn: float = 500.0, cost_fp: float = 200.0):
        """Initialize with merchant unit loss parameters.

        Args:
            cost_fn: Cost of missing an abusive return / wardrobing incident in INR (Lost margin).
            cost_fp: Cost of falsely blocking/delaying a legitimate return in INR (Customer friction & support).
        """
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp

    def evaluate(
        self, y_true: np.ndarray, risk_scores: np.ndarray, output_dir: str = None
    ) -> dict:
        """Execute complete cost-sensitive evaluation and generate plots.

        Args:
            y_true: Ground truth binary labels (1=Abusive return, 0=Legitimate).
            risk_scores: Predicted probabilities / anomaly composite scores.
            output_dir: Directory to save generated PNG plots and JSON summary.

        Returns:
            Dictionary containing metrics, optimal thresholds, and comparison.
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        thresholds = np.arange(0.01, 1.0, 0.01)
        curve_data = []

        best_cost = float("inf")
        opt_cost_result = None

        best_f1 = -1.0
        opt_f1_result = None

        for t in thresholds:
            preds = (risk_scores >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

            total_cost = fp * self.cost_fp + fn * self.cost_fn
            prec = precision_score(y_true, preds, zero_division=0)
            rec = recall_score(y_true, preds, zero_division=0)
            f1 = f1_score(y_true, preds, zero_division=0)

            # Net loss avoided (TP * avg margin saved - FP * friction cost)
            net_benefit = tp * self.cost_fn - fp * self.cost_fp

            res = {
                "threshold": round(float(t), 2),
                "total_cost": float(total_cost),
                "fp_cost": float(fp * self.cost_fp),
                "fn_cost": float(fn * self.cost_fn),
                "net_benefit": float(net_benefit),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1": round(float(f1), 4),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
            curve_data.append(res)

            if total_cost < best_cost:
                best_cost = total_cost
                opt_cost_result = res

            if f1 > best_f1:
                best_f1 = f1
                opt_f1_result = res

        roc_auc = float(roc_auc_score(y_true, risk_scores))
        pr_auc = float(average_precision_score(y_true, risk_scores))

        report = {
            "metrics": {
                "precision": opt_cost_result["precision"],
                "recall": opt_cost_result["recall"],
                "f1": opt_cost_result["f1"],
                "roc_auc": round(roc_auc, 4),
                "pr_auc": round(pr_auc, 4),
            },
            "optimal_threshold": opt_cost_result["threshold"],
            "threshold_selection_method": "Cost Minimization (Total ₹ Loss)",
            "cost_at_optimal": opt_cost_result["total_cost"],
            "cost_params": {"cost_fn": self.cost_fn, "cost_fp": self.cost_fp},
            "confusion_matrix_at_optimal": {
                "tp": opt_cost_result["tp"],
                "fp": opt_cost_result["fp"],
                "tn": opt_cost_result["tn"],
                "fn": opt_cost_result["fn"],
            },
            "comparison": {
                "cost_optimal_threshold": opt_cost_result["threshold"],
                "cost_at_cost_optimal": opt_cost_result["total_cost"],
                "f1_optimal_threshold": opt_f1_result["threshold"],
                "cost_at_f1_optimal": opt_f1_result["total_cost"],
                "financial_savings_vs_f1_max": round(
                    opt_f1_result["total_cost"] - opt_cost_result["total_cost"], 2
                ),
            },
        }

        if output_dir:
            self._save_plots(y_true, risk_scores, curve_data, opt_cost_result, output_dir)
            with open(os.path.join(output_dir, "evaluation_report.json"), "w") as f:
                json.dump(report, f, indent=2)

        return report

    def _save_plots(self, y_true, scores, curve_data, opt_res, output_dir):
        """Generate high-resolution publication charts."""
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except Exception:
            plt.style.use("ggplot")

        # 1. Cost Curve Plot (The Star Metric)
        thresholds = [d["threshold"] for d in curve_data]
        fp_costs = [d["fp_cost"] for d in curve_data]
        fn_costs = [d["fn_cost"] for d in curve_data]
        total_costs = [d["total_cost"] for d in curve_data]

        plt.figure(figsize=(10, 6))
        plt.stackplot(
            thresholds,
            fp_costs,
            fn_costs,
            labels=["FP Friction Cost (Legitimate Returns)", "FN Margin Loss (Missed Abuse)"],
            colors=["#3b82f6", "#ef4444"],
            alpha=0.6,
        )
        plt.plot(thresholds, total_costs, "k-", linewidth=2.5, label="Total Financial Cost")
        plt.axvline(
            opt_res["threshold"],
            color="#10b981",
            linestyle="--",
            linewidth=2,
            label=f"Cost-Optimal Threshold: {opt_res['threshold']:.2f}",
        )
        plt.scatter(
            [opt_res["threshold"]],
            [opt_res["total_cost"]],
            color="#10b981",
            s=120,
            zorder=5,
            edgecolors="black",
        )
        plt.xlabel("Decision Threshold (Risk Score)", fontsize=11)
        plt.ylabel("Total Expected Loss (₹)", fontsize=11)
        plt.title(
            f"Cost-Optimized Threshold Analysis\n(FN Cost = ₹{self.cost_fn:.0f} | FP Cost = ₹{self.cost_fp:.0f})",
            fontsize=13,
            fontweight="bold",
        )
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cost_curve.png"), dpi=150)
        plt.close()

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = roc_auc_score(y_true, scores)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (Held-Out Test)", fontweight="bold")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
        plt.close()

        # 3. Precision-Recall Curve
        prec, rec, _ = precision_recall_curve(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)
        plt.figure(figsize=(7, 5))
        plt.plot(rec, prec, color="#dc2626", lw=2, label=f"PR Curve (AP = {pr_auc:.4f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve (Held-Out Test)", fontweight="bold")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=150)
        plt.close()
