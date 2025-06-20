import pickle
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from joblib import dump
from pathlib import Path
from processing_data import load_data, preprocess_data



def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    models = {
        'logistic_regression': (
            LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
            {'C': [0.1, 1.0, 10.0]}
        ),
        'lightgbm': (
            lgb.LGBMClassifier(class_weight='balanced', random_state=42),
            {'num_leaves': [31, 50], 'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]}
        ),
        'xgboost': (
            xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=1.0, random_state=42),
            {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1]}
        )
    }

    best_model = None
    best_model_name = None
    best_roc_auc = 0.0

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Adjust class imbalance ratio for xgboost
    xgb_ratio = (y_resampled == 0).sum() / (y_resampled == 1).sum()
    models['xgboost'][0].set_params(scale_pos_weight=xgb_ratio)

    for name, (model, param_grid) in models.items():
        print(f"\nTraining and evaluating {name}...")
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', verbose=1, n_jobs=-1)
        grid_search.fit(X_resampled, y_resampled)
        best_params = grid_search.best_params_
        trained_model = grid_search.best_estimator_

        y_pred_proba = trained_model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"{name} - Best Parameters: {best_params}")
        print(f"{name} - Test ROC AUC: {roc_auc:.4f}")

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model = trained_model
            best_model_name = name

    print(f"\nBest Model: {best_model_name} with Test ROC AUC: {best_roc_auc:.4f}")
    return best_model, best_model_name


def save_model_and_scaler(model, model_name, scaler, encoder):
    model_dir = Path("Credit Risk Prediction Model for Fintech Startup") / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"best_{model_name}_model.pkl"
    joblib_path = model_dir / "best_model.joblib"
    scaler_path = model_dir / "scaler.pkl"
    encoder_path = model_dir / "encoder.pkl"

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Pickle model saved to {model_path}")

    dump(model, joblib_path)
    print(f"Joblib model saved to {joblib_path}")

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")

    with open(encoder_path, 'wb') as f:
        pickle.dump(encoder, f)
    print(f"Encoder saved to {encoder_path}")


if __name__ == "__main__":
    print("Starting training pipeline...")

    file_path = r"C:\Users\pc\Downloads\analyzeabtestresults-2\AnalyzeABTestResults 2\Credit Risk Prediction Model for Fintech Startup\scripts\german_credit_data.csv"
    df = load_data(file_path)
    X_train, X_test, y_train, y_test, scaler, encoder = preprocess_data(df)

    best_model, best_model_name = train_and_evaluate_models(X_train, y_train, X_test, y_test)

    save_model_and_scaler(best_model, best_model_name, scaler, encoder)

    print("Training pipeline completed successfully.")
