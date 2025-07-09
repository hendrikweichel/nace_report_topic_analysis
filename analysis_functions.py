import matplotlib.pyplot as plt
import numpy as np

def plot_similarity_distributions(df, cos_threshold: float, NACE_code: str = None, name: str = ""):

    fig, axes = plt.subplots(4, 5, figsize=(20, 12))
    axes = axes.flatten()

    df_scores = df[[column for column in df.columns if "Scores" in column]]

    min_x = df_scores.min().min()
    max_x = df_scores.max().max()

    for i in range(19):
        axes[i].hist(df_scores.iloc[:,i][df_scores.iloc[:,i] > cos_threshold])
        axes[i].set_title(df_scores.iloc[:,i].name[:35])
        axes[i].set_xlim([min_x, max_x])
        axes[i].set_xlabel("cos similarity")
        axes[i].set_ylabel("# chunks")

    if NACE_code: 
        fig.suptitle(f"{name}.csv: NACE class {NACE_code}; cos threshold: {cos_threshold}", fontsize=16)
    else: 
        fig.suptitle(f"{name}.csv; cos threshold: {cos_threshold}", fontsize=16)

    plt.tight_layout()
    return fig

def plot_mean_scores(df, cos_threshold: float, NACE_code: str = None, name: str = ""):

    df_temp = df.iloc[:, 3:][df.iloc[:, 3:] > cos_threshold]
    df_temp = df_temp.fillna(0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df_temp.columns, df_temp.mean())
    ax.set_xticklabels(df_temp.columns, rotation=90)
    
    ax.set_ylabel(f"mean cos. similarity (with threshold {cos_threshold})")
    
    if NACE_code: 
        fig.suptitle(f"{name}.csv: NACE class {NACE_code}; cos threshold: {cos_threshold}", fontsize=16)
    else: 
        fig.suptitle(f"{name}.csv; cos threshold: {cos_threshold}", fontsize=16)
    return fig

def plot_nbr_threshold(df, cos_threshold: float, NACE_code: str = None, name: str = ""):

    df_temp = df.iloc[:, 3:][df.iloc[:, 3:] > cos_threshold]
    non_nan_counts = df_temp.count()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df_temp.columns, non_nan_counts)
    ax.set_xticklabels(df_temp.columns, rotation=90)

    ax.set_ylabel(f"number of chunks cos simliarty > {cos_threshold}")

    if NACE_code: 
        fig.suptitle(f"{name}.csv: NACE class {NACE_code}; cos threshold: {cos_threshold}", fontsize=16)
    else: 
        fig.suptitle(f"{name}.csv; cos threshold: {cos_threshold}", fontsize=16)
    return fig


def plot_correlation_over_document_pages(df, cos_threshold: float = None, NACE_code: str = None, name: str = ""): 
    

    # get top three names
    df_temp = df.iloc[:, 3:][df.iloc[:, 3:] > cos_threshold]
    top3_cols = df_temp.fillna(0).mean().sort_values(ascending=False).head(3)

    fig, ax = plt.subplots(figsize=(12, 6))
    scores = [column for column in df.columns if "Scores" in column]
    for score in top3_cols.index: 
        ax.plot(df_temp[score], label=score)
    
    ax.set_xlabel("Nbr Chunk")
    ax.set_ylabel("Cosine similarity")
    ax.legend()
    if NACE_code: 
        fig.suptitle(f"{name}.csv: NACE class {NACE_code}; cos threshold: {cos_threshold}", fontsize=16)
    else: 
        fig.suptitle(f"{name}.csv; cos threshold: {cos_threshold}", fontsize=16)
    return fig

