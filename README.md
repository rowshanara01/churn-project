# Customer Churn Prediction (Advanced ML + API)

## Chalanor niyom
pip install -r requirements.txt
python generate_data.py      # synthetic data (missing value + imbalance soho)
python train.py              # compare -> tune -> threshold -> explain -> save
pytest -q                    # API test
uvicorn app.main:app --reload   # localhost:8000/docs

## Ki ki shikhbe (file onujayi)
- train.py: ColumnTransformer, Pipeline (data leakage ruke), imputation, one-hot, Stratified K-Fold,
  model comparison, RandomizedSearchCV, class imbalance, ROC-AUC/PR-AUC, cost-based threshold,
  permutation importance
- app/main.py: Pydantic validation (Literal/Field), batch predict, risk level, /model-info
- tests/test_api.py: ML API test kora
- Dockerfile: build er somoy train + test chole

## Nijer jonno challenge
1. generate_data.py er hidden rule bodlao (non-linear interaction dao) -> hist_gb jite kina dekho
2. SMOTE ba undersampling try koro (imbalanced-learn)
3. SHAP diye per-customer explanation /explain endpoint e dao
4. MLflow diye experiment tracking
5. Real dataset: Kaggle "Telco Customer Churn" diye replace koro
6. Drift monitoring: notun data r training data er distribution compare koro
