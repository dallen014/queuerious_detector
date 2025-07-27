"""Visualization and EDA functions for the queuerious_detector project."""

from typing import Optional, List
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer,
    ENGLISH_STOP_WORDS,
)

sns.set_style("dark")


def plot_distribution_with_table(
    df: pd.DataFrame, x: str, y: str, title: str = "Distribution Plot"
) -> None:
    """
    Plots a count distribution of x grouped by y, and prints a summary table
    with counts and percentages for each y category.

    Args:
    - df (pd.DataFrame): The input dataframe
    - x (str): Column representing the measurement (e.g., 'tickets')
    - y (str): Categorical column to group by (e.g., 'queue')
    - title (str): Plot title (optional). If None, a default title is used.
    """

    # Compute counts and percentages
    y_counts = df[y].value_counts()
    y_percent = df[y].value_counts(normalize=True) * 100

    plt.figure(figsize=(12, 6))
    ax = sns.countplot(data=df, y=y, order=y_counts.index)

    # Label each bar with count and percentage
    for i, patch in enumerate(ax.patches):
        count = int(patch.get_width())
        category = y_counts.index[i]
        percent = y_percent[category]
        label = f"{count}\n({percent:.1f}%)"
        ax.text(
            patch.get_width() + 50,  # position a bit right of the bar
            patch.get_y() + patch.get_height() / 2,
            label,
            va="center",
        )

    plt.title(title, loc="center")
    plt.xlabel(f"# of {x}")
    plt.ylabel(y.title())
    plt.tight_layout()
    plt.show()


def plot_queue_heatmap(
    df: pd.DataFrame, x: str, y: str, title: str = "Heatmap Plot"
) -> None:
    """
    Plots a heatmap of ticket counts by queue and priority level.

    Args:
    - df (pd.DataFrame): A crosstab or pivot table DataFrame with counts.
    - x (str): Column name for the x-axis
    - y (str): Column name for the y-axis
    - title (str): Title of the plot.
    """
    ct = pd.crosstab(df[y], df[x])

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        ct, annot=True, fmt="d", cmap="coolwarm", cbar_kws={"label": f"{x} Count"}
    )
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.show()


def plot_text_length_by_queue(
    df: pd.DataFrame,
    text_col: str,
    queue_col: str,
    bins: int = 30,
    title: str = "Histogram Plot",
) -> None:
    """
    Visualizes distribution of text length (word or character count) by queue.

    Args:
        df (pd.DataFrame): Input DataFrame.
        text_col (str): Column name containing the text.
        queue_col (str): Column name representing the queue.
        bins (int): Number of histogram bins.
        title (str): Optional plot title.
    """
    df = df.copy()
    df["text_length"] = df[text_col].fillna("").str.split().apply(len)

    queues = df[queue_col].dropna().unique()
    n = len(queues)
    cols = 3
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows, cols, figsize=(5 * cols, 4 * rows), sharex=True, sharey=True
    )
    axes = axes.flatten()

    for idx, queue in enumerate(queues):
        lengths = df[df[queue_col] == queue]["text_length"]
        sns.histplot(
            df[df[queue_col] == queue]["text_length"],
            bins=bins,
            ax=axes[idx],
            kde=False,
        )
        axes[idx].set_title(queue)
        axes[idx].set_xlabel("Word Count")
        axes[idx].set_ylabel("Frequency")
        axes[idx].tick_params(labelbottom=True)

        # Compute stats
        mean_len = lengths.mean()
        median_len = lengths.median()
        min_len = lengths.min()
        max_len = lengths.max()

        # Annotate in upper-right
        stats_text = f"Mean: {mean_len:.1f}\nMedian: {median_len}\nMin: {min_len}\nMax: {max_len}"
        axes[idx].text(
            0.95,
            0.95,
            stats_text,
            transform=axes[idx].transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
        )

    # Hide any unused subplots
    for j in range(len(queues), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def top_keywords_per_queue(
    df: pd.DataFrame,
    text_col: str,
    queue_col: str,
    method: str = "tfidf",
    top_n: int = 10,
    compare_queues: Optional[List[str]] = None,
    visualize: bool = True,
    extra_stopwords: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Returns a ranked list of top keywords per queue.

    Args:
        df (pd.DataFrame): Input DataFrame.
        text_col (str): Name of the text column.
        queue_col (str): Name of the queue/category column.
        method (str): 'tfidf' or 'count'.
        top_n (int): Number of top words to return per queue.
        compare_queues (List[str], optional): Specific queues to visualize.
        visualize (bool): Whether to plot the keyword comparisons.
        extra_stopwords (List[str], optional): Additional stopwords to exclude.

    Returns:
        pd.DataFrame: DataFrame with top keywords and scores per queue.
    """
    # Build final stopword list
    stopwords_set = set(ENGLISH_STOP_WORDS)
    if extra_stopwords:
        stopwords_set |= set(extra_stopwords)
    stopwords_list = list(stopwords_set)

    df = df[[queue_col, text_col]].dropna()
    results = []
    queues_to_analyze = compare_queues if compare_queues else df[queue_col].unique()

    for queue in queues_to_analyze:
        texts = df.loc[df[queue_col] == queue, text_col].astype(str).tolist()
        if not texts:
            continue

        if method == "tfidf":
            vectorizer = TfidfVectorizer(stop_words=stopwords_list, max_features=1000)
        elif method == "count":
            vectorizer = CountVectorizer(stop_words=stopwords_list, max_features=1000)
        else:
            raise ValueError("method must be 'tfidf' or 'count'")

        try:
            X = vectorizer.fit_transform(texts)
        except ValueError:
            # skip this queue if no valid vocab
            continue

        scores = X.toarray().mean(axis=0)
        vocab = vectorizer.get_feature_names_out()
        top = sorted(zip(vocab, scores), key=lambda x: -x[1])[:top_n]

        for word, score in top:
            results.append({"queue": queue, "word": word, "score": score})

    keyword_df = pd.DataFrame(results)

    if visualize and compare_queues and not keyword_df.empty:
        n = len(queues_to_analyze)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
        if n == 1:
            axes = [axes]

        for ax, queue in zip(axes, queues_to_analyze):
            sub = keyword_df[keyword_df["queue"] == queue].sort_values(
                "score", ascending=False
            )
            if sub.empty:
                ax.set_title(f"{queue}\n(no terms)")
                ax.axis("off")
                continue

            sns.barplot(data=sub, x="score", y="word", color="steelblue", ax=ax)
            ax.set_title(f"Top {top_n} Keywords\n({queue})", fontsize=12)
            ax.set_xlabel("Score")
            ax.set_ylabel("")

            for patch in ax.patches:
                w = patch.get_width()
                ax.text(
                    w + 0.002,
                    patch.get_y() + patch.get_height() / 2,
                    f"{w:.3f}",
                    va="center",
                )

        plt.tight_layout()
        plt.show()
    return keyword_df
