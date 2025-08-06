"""interpretaiblity functions for queuerious_detector"""

from sklearn.neighbors import NearestNeighbors
import numpy as np
from collections import Counter
import re
from typing import List, Dict, Any


def show_nearest_tickets(
    query_index: int,
    X_query: np.ndarray,
    X_ref: np.ndarray,
    df_query: pd.DataFrame,
    df_ref: pd.DataFrame,
    n_neighbors: int = 5,
    top_n: int = 3,
    text_col: str = "redacted_text_clean",
    label_col: str = "queue_grouped",
    preview_len: int = 100,
) -> None:
    """
    Find and display the top-N most similar tickets to a given query ticket using embeddings.

    This function uses NearestNeighbors with cosine distance to identify the closest tickets
    in a reference set (e.g., training) for a given ticket (by index) from a query set (e.g., test).

    Args
    query_index (int): Index of the query ticket in the query DataFrame.
    X_query (np.ndarray): Embedding matrix for the query set (e.g., X_test).
    X_ref (np.ndarray):Embedding matrix for the reference set (e.g., X_train).
    df_query (pd.DataFrame): DataFrame containing the query tickets (e.g., test_df).
    df_ref (pd.DataFrame): DataFrame containing the reference tickets (e.g., train_df).
    n_neighbors (int): Number of nearest neighbors to search for (default is 5).
    top_n (int): Number of top tickets to display, sorted from least to most similar (default is 3).
    text_col (str): Column name containing the ticket text (default is "redacted_text_clean").
    label_col (str): Column name containing the ticket label/class (default is "queue_grouped").
    preview_len (str): Number of characters to show in text preview (default is 100).

    Returns
        None: This function does not return a value; it prints the results directly.
    """
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn_model.fit(X_ref)

    query_embedding = X_query[query_index].reshape(1, -1)
    distances, indices = nn_model.kneighbors(query_embedding)

    print("QUERY TICKET:")
    print(df_query.iloc[query_index][text_col])
    print(
        f"Top {top_n} Most Similar Tickets in Reference Set (Sorted from Least to Most Similar):"
    )

    # Get the top_n, sorted from least to most similar (largest to smallest distance)
    topn = list(zip(distances[0], indices[0]))[:top_n]
    topn = sorted(topn, key=lambda x: -x[0])

    for i, (dist, idx) in enumerate(topn, 1):
        print(f"Neighbor #{i}")
        print(f"Distance: {dist:.3f}")
        print(f"True Label: {df_ref.iloc[idx][label_col]}")
        print("Text Preview:")
        print(df_ref.iloc[idx][text_col][:preview_len])
