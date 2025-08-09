from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import numpy as np
from joblib import load as jload
from sklearn.metrics import classification_report

from queuerious_detector.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR

app = typer.Typer()

# lg model predicting
def predict_lg():
    # loading in data+model+embeddings
    le = jload(MODELS_DIR / "label_encoder.joblib")
    val_df  = pd.read_csv(PROCESSED_DATA_DIR / "val_tickets.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test_tickets.csv")
    y_val  = le.transform(val_df["queue_grouped"])
    y_test = le.transform(test_df["queue_grouped"])
    X_val  = jload(PROCESSED_DATA_DIR / "X_val_tfidf.joblib")
    X_test = jload(PROCESSED_DATA_DIR / "X_test_tfidf.joblib")
    clf = jload(MODELS_DIR / "logreg.joblib")
    # Predicting
    y_val_pred  = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)
    # Creating the reports, printing to console, then saving to a file
    val_report  = classification_report(y_val, y_val_pred, target_names=le.classes_)
    test_report = classification_report(y_test, y_test_pred, target_names=le.classes_)
    print("Validation Classification Report For Logistic Regression:\n", val_report)
    print("Test Classification Report For Logistic Regression:\n", test_report)
    with open(REPORTS_DIR / "logreg_classification_report.txt", "w") as f:
        f.write("Validation Classification Report For Logistic Regression:\n")
        f.write(val_report + "\n\n")
        f.write("Test Classification Report For Logistic Regression:\n")
        f.write(test_report + "\n")

# rf model predicting
def predict_rf():
    # loading in data+model+embeddings
    le = jload(MODELS_DIR / "label_encoder.joblib")
    val_df  = pd.read_csv(PROCESSED_DATA_DIR / "val_tickets.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test_tickets.csv")
    y_val  = le.transform(val_df["queue_grouped"])
    y_test = le.transform(test_df["queue_grouped"])
    X_val  = np.load(PROCESSED_DATA_DIR / "X_val_sbert.npy")
    X_test = np.load(PROCESSED_DATA_DIR / "X_test_sbert.npy")
    clf = jload(MODELS_DIR / "rf.joblib")
    # Predicting
    y_val_pred  = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)
    # Creating the reports, printing to console, then saving to a file
    val_report  = classification_report(y_val, y_val_pred, target_names=le.classes_)
    test_report = classification_report(y_test, y_test_pred, target_names=le.classes_)
    print("Validation Classification Report For Random Forest:\n", val_report)
    print("Test Classification Report For Random Forest:\n", test_report)
    with open(REPORTS_DIR / "rf_classification_report.txt", "w") as f:
        f.write("Validation Classification Report For Random Forest:\n")
        f.write(val_report + "\n\n")
        f.write("Test Classification Report For Random Forest:\n")
        f.write(test_report + "\n")

# svm model predicting
def predict_svm():
    # loading in data+embeddings+model
    le = jload(MODELS_DIR / "label_encoder.joblib")
    val_df  = pd.read_csv(PROCESSED_DATA_DIR / "val_tickets.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test_tickets.csv")
    y_val  = le.transform(val_df["queue_grouped"])
    y_test = le.transform(test_df["queue_grouped"])
    X_val  = np.load(PROCESSED_DATA_DIR / "X_val_sbert.npy")
    X_test = np.load(PROCESSED_DATA_DIR / "X_test_sbert.npy")
    # Need to scale the validation+test sets
    scaler = jload(MODELS_DIR / "svm_scaler.joblib")
    X_val_scaled  = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    clf = jload(MODELS_DIR / "svm.joblib")
    # Predicting
    y_val_pred  = clf.predict(X_val_scaled)
    y_test_pred = clf.predict(X_test_scaled)
    # Creating the reports, printing to console, then saving to a file
    val_report  = classification_report(y_val, y_val_pred, target_names=le.classes_)
    test_report = classification_report(y_test, y_test_pred, target_names=le.classes_)
    print("Validation Classification Report For Standard Vector Classifier:\n", val_report)
    print("Test Classification Report For Standard Vector Classifier:\n", test_report)
    with open(REPORTS_DIR / "svm_classification_report.txt", "w") as f:
        f.write("Validation Classification Report For Standard Vector Classifier:\n")
        f.write(val_report + "\n\n")
        f.write("Test Classification Report Standard Vector Classifier:\n")
        f.write(test_report + "\n")

# xgb model predicting
def predict_xgb():
    # loading in data+embeddings+model
    le = jload(MODELS_DIR / "label_encoder.joblib")
    val_df  = pd.read_csv(PROCESSED_DATA_DIR / "val_tickets.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test_tickets.csv")
    y_val  = le.transform(val_df["queue_grouped"])
    y_test = le.transform(test_df["queue_grouped"])
    X_val  = np.load(PROCESSED_DATA_DIR / "X_val_sbert.npy")
    X_test = np.load(PROCESSED_DATA_DIR / "X_test_sbert.npy")
    clf = jload(MODELS_DIR / "xgb.joblib")
    # Predicting
    y_val_pred  = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)
    # Creating the reports, printing to console, then saving to a file
    val_report  = classification_report(y_val, y_val_pred, target_names=le.classes_)
    test_report = classification_report(y_test, y_test_pred, target_names=le.classes_)
    print("Validation Classification Report For XGBoost:\n", val_report)
    print("Test Classification Report For XGBoost:\n", test_report)
    with open(REPORTS_DIR / "xgb_classification_report.txt", "w") as f:
        f.write("Validation Classification Report For XGBoost:\n")
        f.write(val_report + "\n\n")
        f.write("Test Classification Report For XGBoost:\n")
        f.write(test_report + "\n")


@app.command()

def main(model: str = typer.Option(..., help="'lg' | 'rf' | 'svm' | 'xgb' | 'all'")):
    # if block to determine which model to predict for
    if model == 'lg':
        predict_lg()
    elif model == 'rf':
        predict_rf()
    elif model == 'svm':
        predict_svm()
    elif model == 'xgb':
        predict_xgb()
    elif model == 'all':
         predict_lg()
         predict_rf()
         predict_svm()
         predict_xgb()
    else: 
        logger.error("Unknown option. Options are: 'lg' | 'rf' | 'svm' | 'xgb' | 'all'")


if __name__ == "__main__":
    app()
