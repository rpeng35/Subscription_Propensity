# Merchant Subscription Propensity

**Goal**: Identify existing Stripe merchants likely to adopt the "Subscriptions" product to prioritize sales outreach.

## Key Results:

### Model Performance

* Random Forest Classifier (AUC: 0.79)

### Key Findings

* Insight: "Platform Tenure" and "Industry Fit" are the strongest predictors of adoption.
* Strategy: Targeted 1,000 "Lookalike" merchants, focusing on the Software sector.

## Technical Approach

### Methodology

1. **ETL**: Cleaned "Ghost IDs" (scientific notation corruption) from raw transaction logs.
2. **EDA**: Used Mann-Whitney U tests to validate tenure differences between segments.
3. **Modeling**: Prevented target leakage by excluding subscription-related features from the training set.

## Folder Structure

```text
Subscription_Propensity/
├── data/
│   └── target_merchants.csv
├── src/
│   ├── etl.py
│   ├── data_prep.py
│   ├── feature_eng.py
│   ├── EDA.py
│   └── model.py
├── report/
│   ├── feature_importance.png (and other visuals)
│   └── Propensity_model_report.pdf (final written report)
├── LICENSE
├── README.md
└── requirements.txt
```

**Note on Data**: The raw transaction data (payments.xlsx, merchants.csv) is not included in this repository for privacy reasons.