## Credit Risk Modeling
Predict loan defaults using machine learning on banking data. Feature engineering, imbalance handling, and model tuning included.

### Features:
- KDE-based EDA + IV/VIF feature selection
- Feature engineering: loan-to-income, delinquency ratio
- Class balancing via SMOTE Tomek
- Logistic Regression, Random Forest, XGBoost
- Optuna hyperparameter tuning
- Model evaluation: ROC/AUC, Gini, KS stat, Decile

### Best Model:
- Logistic Regression (with SMOTE + Optuna)
- AUC: 0.98 | Gini: 0.96 | F1-score: 0.94

---

##  Model Deployment via FastAPI

The final logistic regression model was deployed as a real-time REST API using **FastAPI**.

###  API Features:
- `POST /predict_credit_risk`: Accepts borrower info and returns:
  - `probability`: Likelihood of default
  - `credit_score`: Scaled from 300 (high risk) to 900 (low risk)
  - `rating`: Risk category (Poor, Average, Good, Excellent)

### Sample Input:
```json
{
  "age": 35,
  "income": 400000,
  "loan_amount": 150000,
  "loan_tenure_months": 24,
  "avg_dpd_per_delinquency": 5.2,
  "delinquency_ratio": 25.0,
  "credit_utilization_ratio": 70.0,
  "num_open_accounts": 3,
  "residence_type": "Owned",
  "loan_purpose": "Home",
  "loan_type": "Unsecured"
}
```
```
{
  "probability": 0.182,
  "credit_score": 851,
  "rating": "Excellent"
}
```

### Tools:
- Python, Pandas, Scikit-learn, XGBoost, Optuna, Imbalanced-learn, Seaborn, Joblib
