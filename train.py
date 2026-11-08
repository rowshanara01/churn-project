"""Advanced training: preprocessing -> model comparison -> tuning -> cost-based threshold -> explain -> save"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
COST_FN = 500   # churn hobe bujhte na pare -> customer haralam (boro khoti)
COST_FP = 50    # vul kore discount offer dilam (chhoto khoti)

# ---------- 1. Data ----------
df = pd.read_csv("data/churn.csv")
X, y = df.drop(columns="churn"), df["churn"]
num_cols = X.select_dtypes("number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)

# ---------- 2. Preprocessing (leak-free: pipeline er bhitore) ----------
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])

# ---------- 3. Model comparison (Stratified K-Fold, ROC-AUC) ----------
candidates = {
    "logreg": (LogisticRegression(max_iter=2000, class_weight="balanced"),
               {"clf__C": np.logspace(-2, 2, 20)}),
    "random_forest": (RandomForestClassifier(class_weight="balanced_subsample",
                                             random_state=SEED, n_jobs=-1),
                      {"clf__n_estimators": [200, 400], "clf__max_depth": [4, 6, 8, None],
                       "clf__min_samples_leaf": [1, 5, 10]}),
    "hist_gb": (HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
                {"clf__learning_rate": [0.03, 0.05, 0.1], "clf__max_depth": [2, 3, 4],
                 "clf__max_leaf_nodes": [8, 15, 31], "clf__l2_regularization": [0, 1, 5]}),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
scores = {}
for name, (est, _) in candidates.items():
    pipe = Pipeline([("pre", pre), ("clf", est)])
    s = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    scores[name] = float(s.mean())
    print(f"{name:14s} CV ROC-AUC = {s.mean():.3f} +/- {s.std():.3f}")
best_name = max(scores, key=scores.get)
print("\nBest model:", best_name)

# ---------- 4. Hyperparameter tuning (best model e) ----------
est, grid = candidates[best_name]
search = RandomizedSearchCV(Pipeline([("pre", pre), ("clf", est)]), grid, n_iter=15,
                            cv=cv, scoring="roc_auc", random_state=SEED, n_jobs=-1)
search.fit(X_train, y_train)
model = search.best_estimator_
print("Tuned CV ROC-AUC:", round(search.best_score_, 3), "| params:", search.best_params_)

# ---------- 5. Business-cost based threshold (0.5 magic number na) ----------
oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
thresholds = np.linspace(0.05, 0.95, 91)
def cost(t):
    pred = oof >= t
    fn = ((~pred) & (y_train == 1)).sum()
    fp = (pred & (y_train == 0)).sum()
    return fn * COST_FN + fp * COST_FP
threshold = float(min(thresholds, key=cost))
print(f"Best threshold (min business cost): {threshold:.2f}")

# ---------- 6. Final test evaluation (test data ekbar-i chhuyi) ----------
proba = model.predict_proba(X_test)[:, 1]
pred = proba >= threshold
print("\nTEST ROC-AUC:", round(roc_auc_score(y_test, proba), 3),
      "| PR-AUC:", round(average_precision_score(y_test, proba), 3))
print(classification_report(y_test, pred, target_names=["stay", "churn"]))

# ---------- 7. Explainability: permutation importance ----------
imp = permutation_importance(model, X_test, y_test, scoring="roc_auc",
                             n_repeats=10, random_state=SEED)
importance = dict(sorted(zip(X.columns, imp.importances_mean.round(4).tolist()),
                         key=lambda kv: -kv[1]))
print("Feature importance:", importance)

# ---------- 8. Save ----------
joblib.dump({"model": model, "threshold": threshold, "columns": X.columns.tolist()},
            "model/churn_model.joblib")
json.dump({"best_model": best_name, "cv_scores": scores,
           "test_roc_auc": float(roc_auc_score(y_test, proba)),
           "test_pr_auc": float(average_precision_score(y_test, proba)),
           "threshold": threshold, "feature_importance": importance},
          open("model/metrics.json", "w"), indent=2)
print("\nSaved model/churn_model.joblib + model/metrics.json")
