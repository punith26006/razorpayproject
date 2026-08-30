# 🛡️ AI Risk Manager: Return-Risk Scorer & Chargeback Sentinel
### *Track 02: Stop the merchant losing money to fraud, returns, and chargebacks*

> **Submission for Hackathon Track 02 (AI Risk Manager)**  
> **Author:** Punith Vujja ([@punith26006](https://github.com/punith26006))

---

## 📌 Executive Summary

While transaction fraud gets the headlines, **e-commerce return abuse, wardrobing, and friendly chargebacks quietly consume 3% to 8% of merchant net margins** across Indian retail and digital commerce. 

This project delivers a **dedicated, production-grade Return-Risk Scorer and Chargeback Evidence Auto-Drafter** designed specifically to protect merchants from return policy abuse without hurting legitimate customer trust.

```
Incoming Return Request
          │
          ▼
┌───────────────────────────────────────────────────────────┐
│ Feature Engineering & Behavioral Velocity Engine           │
│ (Return Rate, Days-to-Return, Category Norm, 7D Velocity) │
└─────────────────────────────┬─────────────────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│ Supervised XGBoost (70%)  │   │ Anomaly IsoForest (30%)   │
│ (Trained with imbalance   │   │ (Cold-start / New Account │
│  scale_pos_weight)        │   │  Anomaly Detection)       │
└─────────────┬─────────────┘   └─────────────┬─────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
                ┌───────────────────────────┐
                │ Blended Risk Score (0 - 1)│
                └─────────────┬─────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│ Local SHAP Explainability │   │ Asymmetric Cost Curve     │
│ (Feature Attributions)    │   │ (Minimizes Total ₹ Loss)  │
└─────────────┬─────────────┘   └─────────────┬─────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
       ┌──────────────────────────────────────────────┐
       │ Recommended Advisory Action for Human Review │
       │ (INSTANT_REFUND | MANUAL_REVIEW | FLAG)      │
       └──────────────────────────────────────────────┘
```

---

## 🎯 Key Architectural Pillars

### 1. 🛡️ Focused on One Class of Loss
Rather than trying to solve all generic fraud poorly, this system is deeply specialized in **Return Fraud, Wardrobing, and Chargeback Abuse**:
- **Wardrobing detection**: Fast returns (1–3 days) of high-value items where price deviates substantially from category norm.
- **Serial return velocity**: Burst return patterns in short windows (7-day and 30-day rolling velocities).
- **Cold-start anomaly detection**: Isolation Forest scores accounts with sparse history to catch brand-new burner accounts.

### 2. 📊 Honest Metrics on Held-Out Test Set (Zero Leakage)
- **Customer-ID Group Split**: Train and test datasets are partitioned via `GroupShuffleSplit` by `customer_id`. A customer's transactions appear strictly in train OR test, preventing behavioral data leakage.
- **Measured Metrics on Held-Out Test Set (11,994 transactions, 11,602 unseen customers)**:
  - **Precision:** 97.72%
  - **Recall:** 99.57%
  - **F1-Score:** 98.64%
  - **ROC-AUC:** 0.9983
  - **PR-AUC:** 0.9910

### 3. 💰 Asymmetric Cost-Curve Threshold Optimization
In return abuse detection, standard $F_1$ score maximization is economically flawed because misclassification costs are asymmetric:
- **False Negative (FN) Cost:** **₹500** (Merchant loses margin, restocking cost, and product depreciation on fraudulent returns).
- **False Positive (FP) Cost:** **₹200** (Customer friction, CS ticket load, brand trust loss).

We sweep thresholds $T \in [0.01, 0.99]$ to find $T^*$ that minimizes:
$$\text{Total Cost}(T) = \text{FP}(T) \times ₹200 + \text{FN}(T) \times ₹500$$

> **Optimal Operating Threshold:** $T^* = 0.51$.  
> Minimizing financial loss saves **₹4,500+ more per 10k orders** compared to arbitrary $F_1$-max thresholds ($0.63$).

### 4. 🔍 Local SHAP Explainability
Every prediction produces local SHAP (SHapley Additive exPlanations) factor attributions, enabling human risk analysts to immediately understand *why* a return was flagged (e.g., $+0.28$ from 2-day return window, $+0.24$ from 45% lifetime return rate).

### 5. 🔒 Strictly Defense-Only Compliance
This system satisfies the hackathon's disqualification rule:
- ❌ **No automated chargeback disputes**: Never sends automated disputes to card networks.
- ❌ **No automated refund cancellations**: Never auto-debits or blocks customer payments.
- ✅ **Advisory-Only Dossier Generation**: Auto-drafts structured evidence dossiers (fulfillment proof, address verification, SHAP risk summary) exclusively for human analysts to review.

---

## 📂 Repository Structure

```
razorpayproject/
├── app.py                      # Python ML Microservice (XGBoost + IsoForest + SHAP)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Ignores large raw datasets & node_modules
├── README.md                   # Complete architectural & evaluation documentation
│
├── gateway/                    # 🚀 NestJS / TypeScript Merchant API Gateway
│   ├── package.json            # Node.js dependencies (class-validator, axios)
│   ├── tsconfig.json           # TypeScript configuration
│   └── src/
│       ├── main.ts             # Gateway bootstrap (port 3000)
│       ├── app.module.ts       # Root module
│       ├── returns/            # Return scoring & batch evaluation module
│       ├── disputes/           # Razorpay webhook & evidence drafting module
│       └── metrics/            # Cost-curve & benchmark telemetry module
│
├── ml_pipeline/
│   ├── kaggle_training.py      # Complete Kaggle training pipeline (Run on Kaggle)
│   ├── scorer.py               # Real-time inference engine with SHAP attribution
│   └── cost_curve.py           # Cost-curve evaluation & plot generator
│
├── evidence/
│   └── responder.py            # Defense-only chargeback evidence auto-drafter
│
├── templates/
│   └── dashboard.html          # Sleek web dashboard (Cost-Curve Simulator + Stream Test)
│
├── artifacts/
│   └── models/                 # Evaluation reports and feature mapping
│
└── tests/
    └── test_risk_manager.py    # Automated Python test suite
```

---

## 🚀 Quick Start Guide

### Option A: Run the Full-Stack Microservices (NestJS Gateway + Python ML)

```bash
# 1. Start the Python ML Engine (Terminal 1)
pip install -r requirements.txt
python app.py  # Runs on http://localhost:5000

# 2. Start the NestJS Gateway (Terminal 2)
cd gateway
npm install
npm run start:dev  # Runs on http://localhost:3000
```

The NestJS Gateway exposes enterprise merchant endpoints:
- `POST http://localhost:3000/api/v1/returns/score` (Evaluates return request)
- `POST http://localhost:3000/api/v1/returns/batch` (Evaluates batch of return requests)
- `POST http://localhost:3000/api/v1/disputes/evidence` (Generates defense dossier)
- `POST http://localhost:3000/api/v1/disputes/webhook` (Ingests Razorpay dispute webhooks)
- `GET  http://localhost:3000/api/v1/metrics/cost-curve` (Exposes loss curve metrics)

---

### Option B: Run Python Web Dashboard Directly

```bash
pip install -r requirements.txt
python app.py
```
Open your browser and navigate to **`http://localhost:5000`**.  
The dashboard will run with interactive live scoring, the **Cost-Curve & False-Positive Tradeoff Explorer**, SHAP attributions, and evidence generation.

---

### 2. Train Model Weights on Kaggle

To train the models on the 60,000-record E-Commerce Return Abuse Dataset:

1. Open [Kaggle Notebooks](https://www.kaggle.com/code) -> Click **New Notebook**.
2. Click **+ Add Data** (top right) -> Search for:
   `thedevastator/e-commerce-return-abuse-detection-dataset`
3. Copy and paste the entire script from [`ml_pipeline/kaggle_training.py`](ml_pipeline/kaggle_training.py) into the notebook.
4. Click **Run All**.
5. Once completed, download the generated artifacts from `/kaggle/working/`:
   - `return_xgboost.pkl`
   - `return_isolation_forest.pkl`
   - `return_shap_explainer.pkl`
   - `feature_names.json`
   - `evaluation_report.json`
6. Place them inside `artifacts/models/` in your local project folder and restart `app.py`.

---

## 🔌 API Reference

### `POST /api/return-risk`
Evaluates a single return request for return abuse risk.

**Request Payload:**
```json
{
  "customer_id": "CUST_1042",
  "product_category": "Electronics",
  "product_price": 18999,
  "days_to_return": 2,
  "return_rate_per_customer": 0.45,
  "price_vs_category_norm": 1.6,
  "return_velocity_7d": 3,
  "refund_amount_ratio": 0.36
}
```

**Response:**
```json
{
  "risk_score": 0.84,
  "risk_level": "CRITICAL",
  "xgb_probability": 0.88,
  "anomaly_score": 0.74,
  "threshold": 0.52,
  "above_threshold": true,
  "recommended_action": "FLAG_FOR_INSPECTION",
  "action_description": "High probability of wardrobing / serial abuse. Require manual inspection on physical return.",
  "explanation": {
    "top_factors": [
      { "feature": "days_to_return", "impact": 0.28, "value": 2.0 },
      { "feature": "return_rate_per_customer", "impact": 0.24, "value": 0.45 },
      { "feature": "return_velocity_7d", "impact": 0.18, "value": 3.0 }
    ]
  }
}
```

---

### `POST /api/evidence-summary`
Compiles an evidence package for a disputed return or chargeback (Defense-Only).

**Request Payload:**
```json
{
  "transaction": {
    "transaction_id": "TXN-80491",
    "amount": 18999,
    "currency": "INR",
    "product_category": "Electronics",
    "shipping_address": "Mumbai, MH 400050",
    "billing_address": "Mumbai, MH 400050"
  },
  "customer_history": [
    { "amount": 4500, "is_return": false },
    { "amount": 18999, "is_return": true }
  ]
}
```


