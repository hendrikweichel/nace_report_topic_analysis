from typing import List
import pandas as pd
import os
import sys
import tqdm
import numpy as np
import ast

sys.path.append("..")
from src import text_extraction, create_sentence_nace_code_similarities, analysis_functions

def get_position_of_label(classification: dict, label: str):
    
    if not isinstance(label, str): 
        return None

    label = label.lower().strip()
    position = [key[0] for key in classification.keys()].index(label)
    return position

def shorten_csv(df: pd.DataFrame) -> pd.DataFrame: 
    df = df.drop(columns="Embeddings")

    # For all columns except the first, keep only the 100 highest scores, set the rest to NaN
    for col in df.columns[1:]:
        top_100_idx = df[col].nlargest(100).index
        df.loc[~df.index.isin(top_100_idx), col] = np.nan
    df = df[df.iloc[:,1:].apply(lambda x: not pd.isna(x).all(), axis=1)]
    return df


def test_similarities(reports_path: List[str], preprocess_report: callable, threshold_min_chunk_len, cos_threshold, report_to_nace_class, result_path): 

    recording = [] 

    for report_path in tqdm.tqdm(reports_path): 

        # retrieve chunks
        chunks = preprocess_report(pdf_path=report_path)

        # get similarities
        df_similarities = create_sentence_nace_code_similarities.create_sentence_nace_code_similarities(chunks)

        # remove irrelevant chunks
        df_similarities = df_similarities[df_similarities["Sentences"].apply(lambda x: len(x)>threshold_min_chunk_len)]

        # plot 
        fig1 = analysis_functions.plot_mean_scores(df_similarities, cos_threshold=cos_threshold, NACE_code=report_to_nace_class.get(os.path.basename(report_path)), name=os.path.basename(report_path))
        fig2 = analysis_functions.plot_similarity_distributions(df_similarities, cos_threshold=cos_threshold, NACE_code=report_to_nace_class.get(os.path.basename(report_path)), name=os.path.basename(report_path))
        fig3 = analysis_functions.plot_nbr_threshold(df_similarities, cos_threshold=cos_threshold, NACE_code=report_to_nace_class.get(os.path.basename(report_path)), name=os.path.basename(report_path))    

        # create report folder 
        os.makedirs(os.path.join(result_path, os.path.basename(report_path)), exist_ok=True)

        # store the figures
        fig1.savefig(os.path.join(result_path, os.path.basename(report_path), "mean_scores.png"), bbox_inches="tight")
        fig2.savefig(os.path.join(result_path, os.path.basename(report_path), "similarity_distributions.png"), bbox_inches="tight")
        fig3.savefig(os.path.join(result_path, os.path.basename(report_path), "nbr_threshold.png"), bbox_inches="tight")

        # remove duplicates
        df_similarities = df_similarities.drop_duplicates(subset=["Sentences"])
        scores = [column for column in df_similarities.columns if "Scores" in column]

        # apply threshold on similarities
        df_temp = df_similarities[scores][df_similarities[scores] > cos_threshold]   
        df_temp = df_temp.fillna(0)
        mean_vals = df_temp.mean().sort_values(ascending=False)

        store_sentences_path = os.path.join(result_path, os.path.basename(report_path), "relevant_sentences_" + os.path.basename(report_path))
        os.makedirs(store_sentences_path, exist_ok=True)

        i = 1
        # store the 100 most important chunks of the 5 most relevant sectors (shown with mean)
        for sector in df_temp.mean().sort_values(ascending=False)[:5].index: 
            top_chunks = df_temp[sector].sort_values(ascending=False)[:100]
            top_chunks_text = "\n\n".join([f"Score {round(df_similarities.loc[idx][sector], 3)}\n"+ df_similarities.loc[idx]["Sentences"] for idx in top_chunks.index])
            top_chunks_text = sector + top_chunks_text

            with open(os.path.join(store_sentences_path, str(i) + "_" + sector + ".txt"), "w") as f:
                f.write(top_chunks_text) 
            i += 1

        # store df
        df_similarities.to_csv(os.path.join(result_path, os.path.basename(report_path), os.path.basename(report_path) + "_long.csv"))
        df_short = shorten_csv(df_similarities)
        df_short.to_csv(os.path.join(result_path, os.path.basename(report_path), os.path.basename(report_path) + "_short.csv"))

        # record
        mean_vals_dict = {k[7:]:round(v,3) for k,v in mean_vals.to_dict().items()}
        label = report_to_nace_class.get(os.path.basename(report_path))

        position = get_position_of_label(mean_vals_dict, label)

        recording.append({"name": os.path.basename(report_path), "NACE": label,"mean_values": mean_vals_dict, "position_of_label": position})

        df_recording = pd.DataFrame(recording)
        df_recording.to_csv(os.path.join(result_path, "recordings.csv"))


    return df_recording