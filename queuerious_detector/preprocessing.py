"""
preprosses functions for queuerious_detector
"""

from typing import List, Dict, Any
import pandas as pd
from langdetect import detect


def preprocess_tickets(
    df: pd.DataFrame,
    text_fields: List[str],
    lang_col: str = "detected_language",
    combined_col: str = "combined_text",
    target_col: str = "queue",
    new_target_col: str = "target",
    class_map: Dict[Any, Any] = None,
    output_columns: List[str] = None,
) -> pd.DataFrame:
    """
    Preprocesses the support ticket dataframe:
    - Combines text fields into a single column.
    - Detects language.
    - Filters to English tickets only.
    - Combines classes in target column as specified.
    - Returns only the specified columns in the output.

    Args:
        df (pd.DataFrame): Raw data.
        text_fields (List[str]): List of text column names to combine.
        lang_col (str): Name for the detected language column.
        combined_col (str): Name for the combined text column.
        target_col (str): Original target column to remap.
        new_target_col (str): Name for the remapped target column.
        class_map (Dict): Mapping of old class labels to new.
        output_columns (List[str]): List of columns to retain in the output.

    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """
    df = df.copy()

    # combine text columns
    df[combined_col] = df[text_fields].fillna("").agg(" ".join, axis=1).str.strip()

    # detect language
    df[lang_col] = df[combined_col].apply(lambda x: detect(x))

    # keep only English records
    df = df[df[lang_col] == "en"].copy()

    # remap/combine classes for new target
    if class_map is not None:
        df[new_target_col] = df[target_col].map(class_map).fillna(df[target_col])
    else:
        df[new_target_col] = df[target_col]

    # return only specified columns
    if output_columns is not None:
        available_cols = [col for col in output_columns if col in df.columns]
        df = df[available_cols]

    return df
