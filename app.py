"""AI Risk Manager — Return-Risk Scorer & Chargeback Sentinel.

Flask backend serving real-time inference, cost-curve metrics, and defense documentation.
Hackathon Track 02 Submission.
"""

import logging
import os
from flask import Flask, jsonify, render_template, request
import numpy as np

from evidence.responder import EvidenceResponder
from ml_pipeline.scorer import ReturnRiskScorer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize ML & Evidence Services
scorer = ReturnRiskScorer()
evidence_responder = EvidenceResponder()


@app.route("/")
def index():
    """Render the AI Risk Manager Dashboard."""
    return render_template("dashboard.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "service": "AI Risk Manager - Return-Risk Scorer",
            "model_loaded": scorer.is_loaded,
        }
    )


@app.route("/api/return-risk", methods=["POST"])
def analyze_return_risk():
    """Score a single return request for abuse risk.

    Expected JSON body:
    {
        "customer_id": "CUST_1042",
        "product_category": "Electronics",
        "product_price": 15999,
        "days_to_return": 3,
        "return_rate_per_customer": 0.35,
        "price_vs_category_norm": 1.45,
        "return_velocity_7d": 2,
        "return_velocity_30d": 6,
        "refund_amount_ratio": 0.30
    }
    """
    try:
        payload = request.get_json(force=True)
        if not payload:
            return jsonify({"error": "Missing JSON request body"}), 400

        result = scorer.score(payload)
        result["customer_id"] = payload.get("customer_id", "N/A")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error scoring return request: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/return-risk/batch", methods=["POST"])
def batch_return_risk():
    """Score a batch of return requests."""
    try:
        payload = request.get_json(force=True)
        items = payload.get("returns", [])
        if not items:
            return jsonify({"error": "Expected 'returns' array in JSON"}), 400

        scored = []
        high_risk_count = 0
        total_risk_score = 0.0

        for item in items:
            res = scorer.score(item)
            res["customer_id"] = item.get("customer_id", "N/A")
            scored.append(res)
            if res["risk_level"] in ("HIGH", "CRITICAL"):
                high_risk_count += 1
            total_risk_score += res["risk_score"]

        return jsonify(
            {
                "total_scored": len(scored),
                "high_risk_count": high_risk_count,
                "average_risk_score": round(total_risk_score / len(scored), 4)
                if scored
                else 0.0,
                "results": scored,
            }
        )
    except Exception as e:
        logger.error(f"Batch scoring error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/return-risk/metrics", methods=["GET"])
def get_metrics():
    """Return model validation metrics and cost-minimization parameters."""
    return jsonify(scorer.get_metrics())


@app.route("/api/return-risk/sample-returns", methods=["GET"])
def get_sample_returns():
    """Return realistic pre-loaded return scenarios for dashboard demonstration."""
    sample_data = [
        {
            "return_id": "RET-9041",
            "customer_id": "CUST_1042",
            "product_category": "Electronics",
            "product_price": 28999,
            "days_to_return": 2,
            "return_rate_per_customer": 0.45,
            "risk_score": 0.84,
            "risk_level": "CRITICAL",
            "recommended_action": "FLAG_FOR_INSPECTION",
            "top_factor": "Rapid 2-day return on high-value item",
        },
        {
            "return_id": "RET-8823",
            "customer_id": "CUST_5891",
            "product_category": "Fashion",
            "product_price": 4999,
            "days_to_return": 1,
            "return_rate_per_customer": 0.60,
            "risk_score": 0.91,
            "risk_level": "CRITICAL",
            "recommended_action": "FLAG_FOR_INSPECTION",
            "top_factor": "High repeat return rate (60%)",
        },
        {
            "return_id": "RET-7104",
            "customer_id": "CUST_0387",
            "product_category": "Books",
            "product_price": 699,
            "days_to_return": 18,
            "return_rate_per_customer": 0.05,
            "risk_score": 0.12,
            "risk_level": "LOW",
            "recommended_action": "INSTANT_REFUND",
            "top_factor": "Established customer, low historical returns",
        },
        {
            "return_id": "RET-6452",
            "customer_id": "CUST_7721",
            "product_category": "Clothing",
            "product_price": 7500,
            "days_to_return": 5,
            "return_rate_per_customer": 0.25,
            "risk_score": 0.46,
            "risk_level": "MEDIUM",
            "recommended_action": "ENHANCED_MONITORING",
            "top_factor": "Moderate velocity in last 30 days",
        },
        {
            "return_id": "RET-5190",
            "customer_id": "CUST_9904",
            "product_category": "Electronics",
            "product_price": 49999,
            "days_to_return": 3,
            "return_rate_per_customer": 0.38,
            "risk_score": 0.76,
            "risk_level": "HIGH",
            "recommended_action": "MANUAL_REVIEW",
            "top_factor": "High price deviation & short return window",
        },
    ]
    return jsonify(sample_data)


@app.route("/api/evidence-summary", methods=["POST"])
def generate_evidence_summary():
    """Generate a defense-only evidence dossier for a disputed return or chargeback.

    STRICTLY DEFENSE-ONLY: Outputs structured documentation for human validation.
    """
    try:
        payload = request.get_json(force=True) or {}
        transaction = payload.get("transaction", {})
        customer_history = payload.get("customer_history", [])

        risk_eval = None
        if payload.get("include_risk_score", True):
            risk_eval = scorer.score(
                {
                    "product_price": transaction.get("amount", 2500),
                    "days_to_return": 3,
                    "return_rate_per_customer": 0.35,
                    "price_vs_category_norm": 1.2,
                    "return_velocity_7d": 2,
                    "return_velocity_30d": 5,
                    "refund_amount_ratio": 0.25,
                }
            )

        dossier = evidence_responder.generate_evidence_package(
            transaction=transaction,
            customer_history=customer_history,
            risk_evaluation=risk_eval,
        )
        return jsonify(dossier)
    except Exception as e:
        logger.error(f"Evidence generation error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
