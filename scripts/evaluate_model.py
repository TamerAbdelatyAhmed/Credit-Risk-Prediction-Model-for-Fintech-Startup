import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, RocCurveDisplay
#from preprocess_data import preprocess_data
from processing_data import preprocess_data


def load_model(filename='../models/best_lightgbm_model.pkl'):  # Adjust filename
    """Load the trained model from disk."""
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model using various metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # ROC AUC score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score: {roc_auc:.4f}")

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
    plt.title('ROC Curve')
    plt.show()


def model_explainability(model, X_test):
    """Explain the model predictions using SHAP."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary plot
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    plt.title('SHAP Summary Plot (Bar)')
    plt.show()

    shap.summary_plot(shap_values, X_test)
    plt.title('SHAP Summary Plot')
    plt.show()

    # Force plot for a single prediction
    shap.initjs()
    force_plot = shap.force_plot(explainer.expected_value[1], shap_values[1], X_test.iloc[0])
    shap.save_html("shap_force_plot.html", force_plot)  # More descriptive filename


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, scaler = preprocess_data('../data/german_credit_data.csv')
    model = load_model()  # Load the best model
    evaluate_model(model, X_test, y_test)
    model_explainability(model, X_test)