from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from queuerious_detector.config import PROCESSED_DATA_DIR, MODELS_DIR

app = typer.Typer()


@app.command()
def main(model: str = typer.Option(..., help="Feature+Preprocessing pipeline to run: 'lg' or 'sbert'")):
    # ---- DATA PATHS ----
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train_tickets.csv")
    val_df   = pd.read_csv(PROCESSED_DATA_DIR / "val_tickets.csv")
    test_df  = pd.read_csv(PROCESSED_DATA_DIR / "test_tickets.csv")
    # -----------------------------------------

    logger.info("Feature+Preprocessing Started")
    if model == 'lg':
        # custom stopwords from EDA insights
        custom_stopwords = [
            'analytics', 'assistance', 'brand', 'breach', 'data', 'digital', 'inquiry',
            'integration', 'investment', 'issue', 'issues', 'management', 'marketing',
            'medical', 'performance', 'problem', 'project', 'request', 'saas', 'security',
            'software', 'strategies', 'support', 'tools'
        ]
        all_stopwords = list(ENGLISH_STOP_WORDS.union(custom_stopwords))

        #initialize TfidfVectorizer
        tfidf = TfidfVectorizer(
        max_features=10000,
            ngram_range=(1, 2),
            stop_words=all_stopwords
        )

        # fit the vectorizer
        X_train = tfidf.fit_transform(train_df["redacted_text"])
        X_val = tfidf.transform(val_df["redacted_text"])
        X_test = tfidf.transform(test_df["redacted_text"])  

        # Store the features 
        dump(tfidf, MODELS_DIR / "tfidf.joblib", compress=3)
        dump(X_train, PROCESSED_DATA_DIR / "X_train_tfidf.joblib")
        dump(X_val, PROCESSED_DATA_DIR / "X_val_tfidf.joblib")
        dump(X_test, PROCESSED_DATA_DIR / "X_test_tfidf.joblib")

    elif model == 'sbert':
        trans = SentenceTransformer('all-mpnet-base-v2')
        X_train = trans.encode(train_df['redacted_text'].tolist())
        X_val = trans.encode(val_df['redacted_text'].tolist())
        X_test = trans.encode(test_df['redacted_text'].tolist())

        #save embeddings
        np.save(PROCESSED_DATA_DIR / "X_train_sbert.npy", X_train)
        np.save(PROCESSED_DATA_DIR / "X_val_sbert.npy", X_val)
        np.save(PROCESSED_DATA_DIR / "X_test_sbert.npy", X_test)

    else: 
        logger.error("Unknown option. Use 'lg' or 'sbert'")
        return
    
    logger.success("Features generation complete.")

if __name__ == "__main__":
    app()
