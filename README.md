#  Telco Customer Churn Predictor

This project predicts whether a telecom customer is likely to churn (leave the service). It includes data preprocessing, model training, business-aligned evaluation, and a Streamlit dashboard for interactive prediction.

---

##  Overview

**Goal**: Identify customers at risk of churn to enable targeted retention strategies.

**Dataset**: [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Model Used**: Logistic Regression (interpretable and stakeholder-friendly), optimized using both:
- Standard F1 Score
- Business-weighted Fβ Score (β = 4) prioritizing recall

---

##  Key Results

- **Fβ Score** (β = 4): 0.78
- **Test ROC-AUC**: 0.84
- **Top Predictive Features**:
  - Tenure
  - Contract type (One-year, Two-year)
  - Internet Service (Fiber optic)
  - Monthly Charges

---

##  Business Insight

The model shows that:
- Customers with low tenure and no long-term contract are most likely to churn.
- High-churn risk also correlates with fiber internet usage and high monthly bills.
- Targeting early users with personalized offers or incentives could reduce churn.

---

##  Streamlit App

Launch an interactive app to explore predictions:
```bash
streamlit run src/churn_dashboard.py

