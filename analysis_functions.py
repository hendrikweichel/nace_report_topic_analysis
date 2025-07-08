import matplotlib.pyplot as plt

def plot_similarity_distributions(df, cos_threshold, NACE_codes = None, name: str = ""):
    df = df[df["Sentences"].apply(lambda x: len(x)>10)]
    fig, axes = plt.subplots(4, 5, figsize=(20, 12))
    axes = axes.flatten()

    min_x = df.iloc[:,3:].min().min()
    max_x = df.iloc[:,3:].max().max()

    for i in range(19):
        axes[i].hist(df.iloc[:,3+i][df.iloc[:,3+i] > cos_threshold])
        axes[i].set_title(df.iloc[:,3+i].name[:35])
        axes[i].set_xlim([min_x, max_x])

    if NACE_codes: 
        fig.suptitle(f"{name}.csv: NACE class {NACE_codes[name+'.csv']}", fontsize=16)
    else: 
        fig.suptitle(f"{name}.csv", fontsize=16)

    plt.tight_layout()
    return fig

def plot_mean_scores(df, cos_threshold: float, NACE_codes: dict = None, name: str = ""):
    df = df[df["Sentences"].apply(lambda x: len(x)>10)]
    df_temp = df.iloc[:, 3:][df.iloc[:, 3:] > cos_threshold]
    df_temp = df_temp.fillna(0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df_temp.columns, df_temp.mean())
    ax.set_xticklabels(df_temp.columns, rotation=90)
    if NACE_codes: 
        fig.suptitle(f"{name}.csv: NACE class {NACE_codes[name+'.csv']}", fontsize=16)
    else: 
        fig.suptitle(f"{name}.csv", fontsize=16)
    return fig

def plot_nbr_threshold(df, cos_threshold: float, NACE_codes: dict = None, name: str = ""):
    df = df[df["Sentences"].apply(lambda x: len(x)>10)]
    df_temp = df.iloc[:, 3:][df.iloc[:, 3:] > cos_threshold]
    non_nan_counts = df_temp.count()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df_temp.columns, non_nan_counts)
    ax.set_xticklabels(df_temp.columns, rotation=90)
    if NACE_codes: 
        fig.suptitle(f"{name}.csv: NACE class {NACE_codes[name+'.csv']}", fontsize=16)
    else: 
        fig.suptitle(f"{name}.csv", fontsize=16)
    return fig
