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

### Tools:
- Python, Pandas, Scikit-learn, XGBoost, Optuna, Imbalanced-learn, Seaborn, Joblib
