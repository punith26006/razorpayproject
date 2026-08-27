"""Unit tests for AI Risk Manager (Return-Risk Scorer & Evidence Responder)."""

import unittest
import numpy as np

from evidence.responder import EvidenceResponder
from ml_pipeline.cost_curve import CostCurveEvaluator
from ml_pipeline.scorer import ReturnRiskScorer


class TestReturnRiskScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = ReturnRiskScorer()

    def test_low_risk_return(self):
        sample = {
            "customer_id": "CUST_GOOD",
            "product_price": 599,
            "days_to_return": 20,
            "return_rate_per_customer": 0.05,
            "return_velocity_7d": 0,
            "price_vs_category_norm": 0.8,
            "refund_amount_ratio": 0.05,
        }
        result = self.scorer.score(sample)
        self.assertIn("risk_score", result)
        self.assertIn("risk_level", result)
        self.assertEqual(result["risk_level"], "LOW")
        self.assertEqual(result["recommended_action"], "INSTANT_REFUND")
        self.assertFalse(result["above_threshold"])

    def test_high_risk_wardrobing(self):
        sample = {
            "customer_id": "CUST_WARDROBER",
            "product_price": 25000,
            "days_to_return": 1,
            "return_rate_per_customer": 0.65,
            "return_velocity_7d": 4,
            "price_vs_category_norm": 2.5,
            "refund_amount_ratio": 0.50,
        }
        result = self.scorer.score(sample)
        self.assertGreaterEqual(result["risk_score"], 0.75)
        self.assertIn(result["risk_level"], ["HIGH", "CRITICAL"])
        self.assertTrue(result["above_threshold"])
        self.assertIn("explanation", result)
        self.assertTrue(len(result["explanation"]["top_factors"]) > 0)


class TestEvidenceResponder(unittest.TestCase):
    def setUp(self):
        self.responder = EvidenceResponder()

    def test_evidence_generation(self):
        tx = {
            "transaction_id": "TXN-TEST-1234",
            "amount": 14999,
            "currency": "INR",
            "date": "2026-08-20",
            "shipping_address": "Bangalore, India",
            "billing_address": "Bangalore, India",
        }
        history = [
            {"amount": 5000, "is_return": False},
            {"amount": 14999, "is_return": True},
        ]
        dossier = self.responder.generate_evidence_package(
            transaction=tx, customer_history=history
        )
        self.assertEqual(dossier["defense_mode"], "STRICTLY_DEFENSIVE")
        self.assertEqual(dossier["status"], "PENDING_HUMAN_REVIEW")
        self.assertTrue(dossier["transaction_summary"]["address_match"])
        self.assertIn("formatted_evidence_document", dossier)


class TestCostCurveEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = CostCurveEvaluator(cost_fn=500.0, cost_fp=200.0)

    def test_cost_curve_optimization(self):
        np.random.seed(42)
        y_true = np.array([1] * 100 + [0] * 900)
        # Synthetic scores: positives have higher mean
        scores = np.concatenate([
            np.random.beta(5, 2, 100),
            np.random.beta(2, 5, 900)
        ])
        res = self.evaluator.evaluate(y_true, scores)
        self.assertIn("optimal_threshold", res)
        self.assertIn("cost_at_optimal", res)
        self.assertIn("comparison", res)
        self.assertGreater(res["metrics"]["roc_auc"], 0.70)


if __name__ == "__main__":
    unittest.main()
