from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import json
import pandas as pd
import numpy as np
from joblib import dump
from joblib import load as jload
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

from queuerious_detector.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

# lg model training
def train_lg(train_df):
    # loading in all data
    le = jload(MODELS_DIR / "label_encoder.joblib")
    X_train = jload(PROCESSED_DATA_DIR / "X_train_tfidf.joblib")
    y_train = le.transform(train_df["queue_grouped"])
    # Using the best found parameters (from notebooks)
    clf = LogisticRegression(solver='lbfgs', penalty='l2', max_iter=30000, fit_intercept=False, class_weight='balanced', C=10)
    clf.fit(X_train, y_train)
    dump(clf, MODELS_DIR / "logreg.joblib", compress=3)

# rf model training
def train_rf(train_df):
    le = jload(MODELS_DIR / "label_encoder.joblib")
    X_train = np.load(PROCESSED_DATA_DIR / "X_train_sbert.npy")
    y_train = le.transform(train_df["queue_grouped"])
    # Using the best found parameters (from notebooks)
    rf = RandomForestClassifier(n_estimators=200, min_samples_split= 10,min_samples_leaf= 2, max_features='sqrt', max_depth=60, criterion='gini', 
                                    class_weight='balanced', bootstrap=False)
    rf.fit(X_train, y_train)
    dump(rf, MODELS_DIR / "rf.joblib", compress=3)

# svm model training
def train_svm(train_df): 
    le = jload(MODELS_DIR / "label_encoder.joblib")
    X_train = np.load(PROCESSED_DATA_DIR / "X_train_sbert.npy")
    y_train = le.transform(train_df["queue_grouped"])
    # scaling embeddings
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    dump(scaler, MODELS_DIR / "svm_scaler.joblib")
    # Using the best found parameters (from notebooks)
    svm = SVC(class_weight="balanced", random_state=42, probability=True, kernel='poly', gamma=np.float64(0.1), degree=4, C= np.float64(12.915496650148826))
    svm.fit(X_train_scaled, y_train)
    dump(svm, MODELS_DIR / "svm.joblib", compress=3)

# xgb model training
def train_xgb(train_df): 
    le = jload(MODELS_DIR / "label_encoder.joblib")
    X_train = np.load(PROCESSED_DATA_DIR / "X_train_sbert.npy")
    y_train = le.transform(train_df["queue_grouped"])

    # Handle class imbalance
    sample_weight_train= compute_sample_weight(class_weight="balanced", y=y_train)
    xgb = XGBClassifier(objective="multi:softmax",eval_metric="merror",num_class=len(le.classes_),tree_method="hist",n_jobs=-1,random_state=42,n_estimators = 600,
                            min_child_weight = 4, max_depth = 5, learning_rate = 0.1,gamma= 1, colsample_bytree = 0.7, subsample=0.8,reg_lambda=1.5,reg_alpha=0.05)
    xgb.fit(X_train, y_train,sample_weight=sample_weight_train,verbose=False)
    dump(xgb, MODELS_DIR / "xgb.joblib", compress=3)



@app.command()

def main(model: str = typer.Option(..., help="'lg' | 'rf' | 'svm' | 'xgb' | 'all'")):
    # loading in data
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train_tickets.csv")

    # Encode labels
    le = LabelEncoder()
    le.fit(train_df["queue_grouped"])
    dump(le, MODELS_DIR / "label_encoder.joblib")

    # if block to determine which model to train
    if model == 'lg':
        train_lg(train_df)
    elif model == 'rf':
        train_rf(train_df)
    elif model == 'svm':
        train_svm(train_df)
    elif model == 'xgb':
        train_xgb(train_df)
    elif model == 'all':
         train_lg(train_df)
         train_rf(train_df)
         train_svm(train_df)
         train_xgb(train_df)
    else: 
        logger.error("Unknown option. Options are: 'lg' | 'rf' | 'svm' | 'xgb' | 'all'")


if __name__ == "__main__":
    app()
