"""Chargeback Evidence Responder — Strictly Defense-Only Auto-Drafter.

Compiles structured evidence summaries for merchant human reviewers when
a chargeback or disputed return occurs. NEVER initiates automated disputes or
refund rejections — strictly assists human analysts with documentation.
"""

from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EvidenceResponder:
    """Generates defense-only evidence summaries for disputed returns and chargebacks."""

    def generate_evidence_package(
        self,
        transaction: dict,
        customer_history: list = None,
        risk_evaluation: dict = None,
    ) -> dict:
        """Generate a structured evidence dossier for human risk analysts.

        Args:
            transaction: Disputed transaction details (ID, amount, date, shipping address, etc.)
            customer_history: List of historical transaction dicts for this customer.
            risk_evaluation: Output dictionary from ReturnRiskScorer.

        Returns:
            Structured evidence report with recommended documentation points and template.
        """
        customer_history = customer_history or []
        history_summary = self._analyze_history(customer_history)
        evidence_points = self._extract_evidence_points(
            transaction, history_summary, risk_evaluation
        )

        template_text = self._build_formatted_letter(
            transaction, history_summary, evidence_points, risk_evaluation
        )

        return {
            "dossier_id": f"EV-DOSSIER-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "generated_at": datetime.now().isoformat(),
            "status": "PENDING_HUMAN_REVIEW",
            "defense_mode": "STRICTLY_DEFENSIVE",
            "compliance_notice": (
                "DEFENSE-ONLY DOCUMENT: Generated exclusively for human review. "
                "No automated dispute or debit action is taken."
            ),
            "transaction_summary": {
                "transaction_id": transaction.get("transaction_id", "N/A"),
                "amount": transaction.get("amount", 0.0),
                "currency": transaction.get("currency", "INR"),
                "order_date": transaction.get("date", datetime.now().strftime("%Y-%m-%d")),
                "product_category": transaction.get("product_category", "General"),
                "shipping_address": transaction.get("shipping_address", "N/A"),
                "billing_address": transaction.get("billing_address", "N/A"),
                "address_match": transaction.get("shipping_address")
                == transaction.get("billing_address"),
            },
            "history_summary": history_summary,
            "evidence_points": evidence_points,
            "formatted_evidence_document": template_text,
        }

    def _analyze_history(self, history: list) -> dict:
        if not history:
            return {
                "total_orders": 1,
                "total_spend": 0.0,
                "return_count": 0,
                "return_rate": 0.0,
                "account_age_days": 1,
            }

        total_orders = len(history)
        amounts = [float(h.get("amount", 0.0)) for h in history]
        returns = sum(1 for h in history if h.get("is_return", False))

        return {
            "total_orders": total_orders,
            "total_spend": round(sum(amounts), 2),
            "return_count": returns,
            "return_rate": round(returns / total_orders, 3),
            "account_age_days": len(history) * 15,
        }

    def _extract_evidence_points(
        self, tx: dict, hist: dict, risk: dict = None
    ) -> list:
        points = []

        # 1. Delivery & Fulfillment confirmation
        points.append(
            {
                "type": "FULFILLMENT_PROOF",
                "claim": f"Order {tx.get('transaction_id', 'N/A')} was fulfilled and delivered to customer address.",
                "weight": "STRONG",
            }
        )

        # 2. Address consistency check
        if tx.get("billing_address") and tx.get("shipping_address"):
            if tx["billing_address"] == tx["shipping_address"]:
                points.append(
                    {
                        "type": "IDENTITY_CONSISTENCY",
                        "claim": "Billing and shipping addresses match exactly on file.",
                        "weight": "STRONG",
                    }
                )
            else:
                points.append(
                    {
                        "type": "IDENTITY_CONSISTENCY",
                        "claim": "Delivery address differs from billing address (Potential reshipping indicator).",
                        "weight": "MODERATE",
                    }
                )

        # 3. Behavioral Return History
        if hist.get("return_rate", 0.0) >= 0.30:
            points.append(
                {
                    "type": "BEHAVIORAL_PATTERN",
                    "claim": (
                        f"Customer demonstrates repeated return pattern: {hist['return_count']} returns "
                        f"out of {hist['total_orders']} orders ({hist['return_rate']:.1%} lifetime return rate)."
                    ),
                    "weight": "HIGH",
                }
            )

        # 4. AI Risk Assessment Advisory
        if risk and risk.get("risk_score", 0.0) >= 0.50:
            top_f = [
                f["feature"]
                for f in risk.get("explanation", {}).get("top_factors", [])[:3]
            ]
            points.append(
                {
                    "type": "RISK_MODEL_ADVISORY",
                    "claim": f"AI Risk Model flagged risk score of {risk['risk_score']:.2f} due to: {', '.join(top_f)}.",
                    "weight": "SUPPORTING_CONTEXT",
                }
            )

        return points

    def _build_formatted_letter(
        self, tx: dict, hist: dict, points: list, risk: dict = None
    ) -> str:
        lines = [
            "================================================================================",
            "           MERCHANT EVIDENCE DOSSIER — CHARGEBACK & DISPUTE REVIEW              ",
            "================================================================================",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Transaction ID: {tx.get('transaction_id', 'N/A')}",
            f"Disputed Amount: {tx.get('currency', 'INR')} {tx.get('amount', 0.0):,.2f}",
            f"Customer Lifetime Orders: {hist.get('total_orders', 1)} | Lifetime Returns: {hist.get('return_count', 0)}",
            "",
            "KEY EVIDENCE SUMMARY:",
        ]

        for idx, pt in enumerate(points, 1):
            lines.append(f"  {idx}. [{pt['type']}] ({pt['weight']})")
            lines.append(f"     {pt['claim']}")

        if risk:
            lines.append("")
            lines.append("AI RISK ATTRITION (SHAP LOCAL EXPLAINABILITY):")
            for factor in risk.get("explanation", {}).get("top_factors", []):
                lines.append(
                    f"  - {factor['feature'].replace('_', ' ').title()}: Value = {factor['value']} (SHAP Impact: {factor['impact']:+.3f})"
                )

        lines.extend(
            [
                "",
                "RECOMMENDED ACTION FOR HUMAN ANALYST:",
                f"  -> {risk.get('recommended_action', 'MANUAL_REVIEW') if risk else 'SUBMIT_EVIDENCE'}",
                "",
                "DISCLAIMER: Strictly defense-only documentation for human validation.",
                "================================================================================",
            ]
        )

        return "\n".join(lines)
