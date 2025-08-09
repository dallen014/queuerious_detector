from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd

from queuerious_detector.preprocessing import *
from queuerious_detector.leak_prevent import deduplicate_and_split
from queuerious_detector.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- DATA PATHS ----
    #input_path: Path = RAW_DATA_DIR /  "aa_dataset-tickets-multi-lang-5-2-50-version.csv",
    train_out: Path = PROCESSED_DATA_DIR / "train_tickets.csv",
    val_out: Path = PROCESSED_DATA_DIR / "val_tickets.csv",
    test_out: Path = PROCESSED_DATA_DIR / "test_tickets.csv",
    # ----------------------------------------------
):
    # Loading in all files in the raw folder
    files = list(RAW_DATA_DIR.glob("*.csv"))
    comb = [pd.read_csv(p) for p in files]
    raw_data = pd.concat(comb, ignore_index=True)

    logger.info("Processing dataset...")
    #combine classes based on eda analysis
    class_map = {'Technical Support': 'Technical & IT Support',
        'IT Support': 'Technical & IT Support',
        'Customer Service': 'Customer Service, Returns & Exchanges',
        'Returns and Exchanges': 'Customer Service, Returns & Exchanges'
    }

    #preprocess the data
    preprocessed_df = preprocess_tickets(
        df=raw_data,
        text_fields=["subject", "body"],
        target_col="queue",
        new_target_col="queue_grouped",
        class_map=class_map,
        output_columns=["combined_text", "queue_grouped"]
    )

    #redact PII from the combined text
    preprocessed_df["redacted_text"] = preprocessed_df["combined_text"].apply(redact_pii)

    # clean text
    preprocessed_df["redacted_text_clean"] = preprocessed_df["redacted_text"].apply(clean_text)

    #split and save the data
    train_df, val_df, test_df = deduplicate_and_split(
        preprocessed_df,
        text_col="redacted_text_clean",
        target_col="queue_grouped"
    )

    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    test_df.to_csv(test_out, index=False)

    logger.success("Processing dataset complete.")



if __name__ == "__main__":
    app()
