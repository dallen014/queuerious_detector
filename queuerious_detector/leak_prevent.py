"""data leakage prevention"""

import pandas as pd
from sklearn.model_selection import train_test_split


def deduplicate_and_split(
    df: pd.DataFrame,
    text_col: str = "redacted_text",
    target_col: str = "target",
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple:
    """
    Deduplicate a ticket dataset by removing rows with duplicate text in the specified column.
    Then, split the deduplicated data into training, validation, and test sets, ensuring no overlap between splits.

    Args:
        df (pd.DataFrame): The input DataFrame containing ticket data.
        text_col (str): The name of the column containing ticket text to check for duplicates.
        target_col (str): The name of the target variable column for stratification.
        val_size (float): Proportion of data to include in the validation set (default: 0.15).
        test_size (float): Proportion of data to include in the test set (default: 0.15).
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (train_df, val_df, test_df) - Deduplicated DataFrames for each split.
    """
    # remove duplicate tickets keeping first occurrence
    deduped_df = df.drop_duplicates(subset=[text_col], keep="first").reset_index(
        drop=True
    )

    # first split off the test set
    train_val_df, test_df = train_test_split(
        deduped_df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=deduped_df[target_col],
    )

    # split off validation set
    val_relative_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        random_state=random_state,
        shuffle=True,
        stratify=train_val_df[target_col],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
