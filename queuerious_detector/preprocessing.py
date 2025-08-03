"""
preprocess functions for queuerious_detector
"""

from typing import List, Dict, Any
import pandas as pd
from langdetect import detect
import spacy
import re

nlp_en = spacy.load("en_core_web_lg")


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


def redact_pii(text: Any) -> str:
    """
    Redact PII from text using regex and Named Entity Recognition (NER).

    Regex:
      - Emails
      - Phone numbers
      - IP addresses
      - Credit card numbers
      - Street-style addresses

    NER:
      - PERSON (names)

    Args:
        text (Any): Input text to redact.
        lang (str): Language code ('en' or 'de') for appropriate NER model.

    Returns:
        str: Text with PII replaced by placeholders.
    """
    if not isinstance(text, str):
        return ""

    redacted = text

    # regex patterns for personal identifiable information (PII)
    patterns = {
        "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        "phone": (
            r"\b(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?|\d{3})" r"[-.\s]?\d{3}[-.\s]?\d{4}\b"
        ),
        "ip": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
        "address": r"\b\d{1,5}\s+\w+(?:\s\w+)?\s+(St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane)\b",
    }
    # apply regex patterns to redact PII
    for key, pattern in patterns.items():
        redacted = re.sub(pattern, f"[{key.upper()}_REDACTED]", redacted)

    # NER-based redaction
    doc = nlp_en(redacted)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            redacted = redacted.replace(ent.text, "<NAME>")

    return redacted


def clean_text(text):
    """
    Clean text minimally for transformer-based embeddings.

    Performs minimal cleaning appropriate for transformer models:
      - Strips leading and trailing whitespace.
      - Normalizes internal whitespace to a single space.
      - Removes line breaks.
      - Optionally removes HTML tags.

    Args:
        text (Any):The input text to be cleaned.

    Returns
        str: Cleaned text string .
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()

    # replace multiple whitespace with single space
    text = re.sub(r"\s+", " ", text)

    # remove line breaks
    text = text.replace("\n", " ").replace("\r", " ")
    return text
