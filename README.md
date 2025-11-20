PWR-Abnormality-Detection-RandomForest

This repository contains the full implementation of the machine-learning framework used in the research paper:

The project focuses on detecting bias and drift faults in PWR sensor readings (pressure, power, vibration).
Machine Learning—specifically Random Forest with SMOTE and GridSearchCV optimization—was used to handle noisy data, imbalance, and complex reactor dynamics.

🔥 Key Features

✔️ Handles severe class imbalance using SMOTE
✔️ Hyperparameter optimization using GridSearchCV (F1-score optimized)
✔️ Full evaluation: accuracy, precision, recall, F1-score, AUC
✔️ Confusion matrix, ROC curve, feature importance, learning curve
✔️ Modular and production-ready Python code
✔️ Reproducible experiments linked to your publication

📊 Model Performance (from the research paper)
Metric	Score
Accuracy	96.0%
Precision	0.98
Recall	0.93
F1-Score	0.96
AUC	0.996

These results show superior performance over KNN, Decision Tree, AdaBoost, XGBoost, and LightGBM.

▶️ How to Run
1. Install dependencies
pip install -r requirements.txt

2. Run the main script
python main.py

📘 Methodology Overview

Dataset Preparation

Pressurized Water Reactor dataset (Kaggle)
Link: https://www.kaggle.com/datasets/rahatkhan6/pressurized-water-reactor

Encoding + cleaning

Class Imbalance Solution

Synthetic Minority Oversampling Technique (SMOTE)

Model Optimization

Random Forest with GridSearchCV

5-fold cross-validation

Optimization target: F1-score

Evaluation Metrics

Accuracy, Precision, Recall, F1-score

ROC-AUC

Confusion Matrix

Learning Curve

Feature Importance

Result Interpretation

Vibration sensor features showed highest predictive value

RF significantly outperformed other ML models

🧪 Reproducing Figures

The repository automatically saves:

confusion_matrix.png

roc_curve.png

feature_importance.png

learning_curve.png

These were used directly in your publication.

🤝 License

MIT License.
