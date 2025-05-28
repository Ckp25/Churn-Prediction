import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, make_scorer, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score


def weighted_fbeta(y_true, y_pred, beta=4):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    if precision + recall == 0:
        return 0
    return (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)


def run_training(input_path=r'D:\Customer Churn\data\preprocessed_script_churn_data.csv',
                 model_fbeta_path='models/logistic_model_fbeta.joblib',
                 model_f1_path='models/logistic_model_f1.joblib',
                 test_data_path='data/test_data.csv'):
    # Load preprocessed dataset
    df = pd.read_csv(input_path)
    X = df.drop(columns='Churn')
    y = df['Churn'].map({'No': 0, 'Yes': 1})

    # Split into train, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )

    # Define hyperparameter grid
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'class_weight': ['balanced']
    }

    # F-beta Grid Search
    fbeta_scorer = make_scorer(weighted_fbeta, beta=4)
    grid_fbeta = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid,
        scoring=fbeta_scorer,
        cv=5,
        verbose=1,
        n_jobs=-1
    )
    grid_fbeta.fit(X_train, y_train)
    best_model_fbeta = grid_fbeta.best_estimator_

    # F1 Grid Search
    grid_f1 = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid,
        scoring='f1',
        cv=5,
        verbose=1,
        n_jobs=-1
    )
    grid_f1.fit(X_train, y_train)
    best_model_f1 = grid_f1.best_estimator_

    # Evaluate on held-out test set
    print("\n--- Evaluation: F-beta Tuned Model ---")
    y_pred_fbeta = best_model_fbeta.predict(X_test)
    y_prob_fbeta = best_model_fbeta.predict_proba(X_test)[:, 1]
    print("Test ROC-AUC:", roc_auc_score(y_test, y_prob_fbeta))
    print("Classification Report:\n", classification_report(y_test, y_pred_fbeta))

    print("\n--- Evaluation: F1 Tuned Model ---")
    y_pred_f1 = best_model_f1.predict(X_test)
    y_prob_f1 = best_model_f1.predict_proba(X_test)[:, 1]
    print("Test ROC-AUC:", roc_auc_score(y_test, y_prob_f1))
    print("Classification Report:\n", classification_report(y_test, y_pred_f1))

    # Save both models
    os.makedirs(os.path.dirname(model_fbeta_path), exist_ok=True)
    joblib.dump(best_model_fbeta, model_fbeta_path)
    joblib.dump(best_model_f1, model_f1_path)
    print("\n Models saved to:")
    print("F-beta model:", model_fbeta_path)
    print("F1 model:", model_f1_path)

    # Save test set
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data.to_csv(test_data_path, index=False)
    print(" Held-out test set saved to:", test_data_path)

if __name__ == "__main__":
    run_training()
