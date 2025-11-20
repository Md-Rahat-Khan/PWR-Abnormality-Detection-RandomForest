"""
Main script for Optimized Random Forest–based Abnormality Detection in PWRs
Author: Rehat
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib

from src.evaluation import evaluate_model, plot_confusion_matrix, plot_roc, plot_learning_curve
from src.feature_importance import plot_feature_importance

# ================================
# Load & Encode Dataset
# ================================

df = pd.read_csv("data/Hackatom_Task_2.csv")
encoder = LabelEncoder()
df["Status"] = encoder.fit_transform(df["Status"])

X = df.drop("Status", axis=1)
y = df["Status"]

# ================================
# Apply SMOTE
# ================================

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# ================================
# Train-Test Split
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# ================================
# Random Forest + Grid Search
# ================================

rf = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [2, 5, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False]
}

grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# Save model
joblib.dump(best_model, "models/best_random_forest.pkl")

# ================================
# Evaluation
# ================================

print("Best Params:", grid.best_params_)

pred = best_model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, pred))

cm = confusion_matrix(y_test, pred)
plot_confusion_matrix(cm)

evaluate_model(best_model, X_train, X_test, y_train, y_test)
plot_roc(best_model, X_test, y_test)
plot_learning_curve(best_model, X_res, y_res)

plot_feature_importance(best_model, X_res.columns)
