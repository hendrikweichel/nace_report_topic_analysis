from typing import List
import pandas as pd
import os
import sys
import tqdm
import numpy as np
import ast


sys.path.append("/Users/hendrikweichel/projects/NaceCodeClassification/nace_report_topic_analysis_3")
sys.path.append("..")
from src.text_extraction import text_extraction
from src import create_sentence_nace_code_similarities, analysis_functions

def translate_classification_to_other_level(classification, other_level, df_nace_codes_descriptions=None): 
    """ Take one classificaiton to another level, therefore: 
    1. Translate each classification to the objective level
    2. Take mean value
    
    e.g. {1: 0.2, 2: 0.1, 1: 0.05} -> {A: 0.125, B: 0.1}

    Args:
        classification (_type_): _description_
        other_level (_type_): _description_
    """

    # get code
    one_original_nace_code = list(classification.items())[0][0].split("_")[0]
    original_level = get_nace_level(one_original_nace_code)

    assert original_level >= other_level, "original level: " + str(original_level) +  ">=" + "wanted level: " + str(other_level)

    if df_nace_codes_descriptions is None: 
            df_nace_codes_descriptions = pd.read_csv("../data/NACE_Rev2_Structure_Explanatory_Notes_EN__1_.tsv", sep="\t")

    new_classification = pd.DataFrame(columns=["nace_class", "score"])

    for key, value in classification.items(): 
        
        new_code = get_all_level(key.split("_")[0]).get(other_level, "")
        new_name = new_code + "_" + df_nace_codes_descriptions[df_nace_codes_descriptions["CODE"] == new_code]["NAME"]
        
        if len(new_classification) == 0: 
            new_classification = pd.DataFrame({"nace_class": new_name, "score": value})
        else: 
            new_classification = pd.concat([new_classification, pd.DataFrame({"nace_class": new_name, "score": value})], ignore_index=True)

    new_classification = new_classification.groupby("nace_class").aggregate("mean").sort_values("score", ascending=False).to_dict(orient="dict")["score"]
    return new_classification

def get_nace_level(nace_code):
    if float(nace_code) < 10:
        nace_code = "0" + str(nace_code)
    original_level = len(nace_code.replace(".", ""))
    return original_level

def get_evaluation(classification: dict, label: str, df_nace_codes_descriptions):
    """ In classification and label. Out: 
    - for each level over label: on which position is the level?: 

    Args:
        classification (dict): _description_
        label (str): _description_
    """

    evaluation = dict()

    # get all levels of the label
    all_levels_label = get_all_level(label, df_nace_codes_descriptions)
    
    one_original_nace_code = list(classification.items())[0][0].split("_")[0]
    original_level = get_nace_level(one_original_nace_code)
    
    for level in range(1, min(original_level, max(list(all_levels_label.keys()))) + 1):

        # if original_level != level: 
        #     new_classification = translate_classification_to_other_level(classification, level)
        # else:
        #     new_classification = classification
        new_classification = translate_classification_to_other_level(classification, level)
        print(new_classification)

        try: 
            position = [k.split("_")[0] for k in new_classification].index(str(all_levels_label[level]))
        except ValueError: 
            position = None

        evaluation["position_lvl_" + str(level)] = position
        evaluation["classes_lvl_" + str(level)] = len(new_classification)
        evaluation["classification_lvl_" + str(level)] = new_classification
    
    return evaluation


def get_level_1_nace(x):
    x = float(x)
    
    if 1 <= x <= 3.22:
        return 'A'
    elif 5 <= x <= 9.9:
        return 'B'
    elif 10 <= x <= 33.20:
        return 'C'
    elif 35 <= x <= 35.30:
        return 'D'
    elif 36 <= x <= 39:
        return 'E' 
    elif 41 <= x <= 43.99:
        return 'F'
    elif 45 <= x <= 47.99:
        return 'G'
    elif 49 <= x <= 53.2:
        return 'H'
    elif 55 <= x <= 56.3:
        return 'I'
    elif 58 <= x <= 63.99:
        return 'J'
    elif 64 <= x <= 66.3:
        return 'K'
    elif 68 <= x <= 68.32:
        return 'L'
    elif 69 <= x <= 75:
        return 'M' 
    elif 77 <= x <= 82.99:
        return 'N'
    elif 84 <= x <= 84.3:
        return 'O'
    elif 85 <= x <= 85.6:
        return 'P'
    elif 86 <= x <= 88.99:
        return 'Q'
    elif 90 <= x <= 93.29:
        return 'R'
    elif 94 <= x <= 96.09:
        return 'S' 
    elif 97 <= x <= 98.2:
        return 'T'
    elif x == 99:
        return 'U'
    else:
        return None

def get_all_level(nace_code, df_nace_codes_descriptions= None): 

    if isinstance(nace_code, int) or isinstance(nace_code, float):
        if nace_code < 10:
            nace_code = "0" + str(nace_code)
        
    nace_code = str(nace_code)

    init_level = None 

    if nace_code.isalpha(): 
        init_level = 1
    elif "." not in nace_code: 
        init_level = 2
    else: 
        init_level = len(nace_code.split(".")[1])+2

    levels = {init_level: nace_code}

    for level in range(1,init_level):     
        if level == 3: 
            levels[level] = str(nace_code)[:-1]
        if level == 2: 
            levels[level] = str(int(nace_code[:2]))
        if level == 1: 
            levels[level] = get_level_1_nace(float(nace_code))

    return levels

def get_position_of_label(classification: dict, label: str):

    level = len(list(classification.items())[0][0].split("_")[0].replace(".", ""))

    levels = get_all_level(label)

    label_of_level = levels[level]
    
    position = [key.split("_")[0] for key in classification.keys()].index(str(label_of_level))
    
    return position


def shorten_csv(df: pd.DataFrame) -> pd.DataFrame: 
    if "Embeddings" in df.columns: 
        df = df.drop(columns="Embeddings")

    # For all columns except the first, keep only the 100 highest scores, set the rest to NaN
    for col in df.columns[1:]:
        top_100_idx = df[col].nlargest(100).index
        df.loc[~df.index.isin(top_100_idx), col] = np.nan
    df = df[df.iloc[:,1:].apply(lambda x: not pd.isna(x).all(), axis=1)]
    return df


def test_similarities(reports_path: List[str], preprocess_report: callable, threshold_min_chunk_len, cos_threshold, report_to_nace_class, result_path, level=1, overwrite=True): 

    recording = [] 

    df_nace_codes_descriptions = pd.read_csv("../data/NACE_Rev2_Structure_Explanatory_Notes_EN__1_.tsv", sep="\t")

    for report_path in tqdm.tqdm((reports_path)): 

        label = report_to_nace_class.get(os.path.basename(report_path))
        level_1_label = get_all_level(label)[1]

        #print("Report: ", report_path)

        # create report folder 
        store_result_at = os.path.join(result_path, os.path.basename(report_path))
        os.makedirs(store_result_at, exist_ok=True)

        if overwrite: 
            if os.path.exists(store_result_at + "/" + os.path.basename(report_path) + "_long.csv"):
                continue

        # retrieve chunks
        chunks = preprocess_report(pdf_path=report_path)

        # remove chunks with length smaller than threshold
        chunks = [chunk for chunk in chunks if len(chunk) > threshold_min_chunk_len]
        
        # remove duplicates
        chunks = list(set(chunks))
        
        if len(chunks) == 0:
            continue

        #print("Number of Chunks: ", len(chunks))

        # get similarities
        df_similarities = create_sentence_nace_code_similarities.create_sentence_nace_code_similarities(chunks, level=level, df_nace_codes_descriptions=df_nace_codes_descriptions)

        # # plot 
        # fig1 = analysis_functions.plot_mean_scores(df_similarities, cos_threshold=cos_threshold, NACE_code=report_to_nace_class.get(os.path.basename(report_path)), name=os.path.basename(report_path))
        # fig2 = analysis_functions.plot_similarity_distributions(df_similarities, cos_threshold=cos_threshold, NACE_code=report_to_nace_class.get(os.path.basename(report_path)), name=os.path.basename(report_path))
        # fig3 = analysis_functions.plot_nbr_threshold(df_similarities, cos_threshold=cos_threshold, NACE_code=report_to_nace_class.get(os.path.basename(report_path)), name=os.path.basename(report_path))    

        # # store the figures
        # fig1.savefig(os.path.join(result_path, os.path.basename(report_path), "mean_scores.png"), bbox_inches="tight")
        # fig2.savefig(os.path.join(result_path, os.path.basename(report_path), "similarity_distributions.png"), bbox_inches="tight")
        # fig3.savefig(os.path.join(result_path, os.path.basename(report_path), "nbr_threshold.png"), bbox_inches="tight")

        scores_column_names = [column for column in df_similarities.columns if "Scores" in column]

        # apply threshold on similarities
        df_temp = df_similarities[scores_column_names][df_similarities[scores_column_names] > cos_threshold]   

        # replace na vals with 0
        df_temp = df_temp.fillna(0)

        # get mean values for each nace code
        mean_vals = df_temp.mean().sort_values(ascending=False)

        # make folder 
        store_sentences_path = os.path.join(result_path, os.path.basename(report_path), "relevant_sentences_" + os.path.basename(report_path))
        os.makedirs(store_sentences_path, exist_ok=True)

        # store the 100 most important chunks of the 5 most relevant sectors (shown with mean)
        i = 1
        for sector in df_temp.mean().sort_values(ascending=False)[:5].index: 
            top_chunks = df_similarities[sector].sort_values(ascending=False)[:100]
            top_chunks_text = "\n\n".join([f"Score {round(df_similarities.loc[idx][sector], 3)}\n"+ df_similarities.loc[idx]["Sentences"] for idx in top_chunks.index if df_similarities.loc[idx][sector] > cos_threshold])
            top_chunks_text = sector + "\n\n\n" + top_chunks_text

            with open(os.path.join(store_sentences_path, str(i) + "_" + sector + ".txt"), "w") as f:
                f.write(top_chunks_text) 
            i += 1

        # store df
        df_similarities = df_similarities.drop(columns=["Embeddings"])
        df_similarities.to_csv(os.path.join(result_path, os.path.basename(report_path), os.path.basename(report_path) + "_long.csv"))
        df_short = shorten_csv(df_similarities)
        df_short.to_csv(os.path.join(result_path, os.path.basename(report_path), os.path.basename(report_path) + "_short.csv"))

        # record the results 
        mean_vals_dict = {k[7:]:round(v,3) for k,v in mean_vals.to_dict().items()}
        
        # get label of the report
        label = report_to_nace_class.get(os.path.basename(report_path))
        
        try: 
            if isinstance(label, float) or isinstance(label, int):                     
                if len(str(label).split(".")[0]) == 1: 
                    label = "0" + str(label) 
            label_description = df_nace_codes_descriptions[df_nace_codes_descriptions["CODE"] == str(label)]["NAME"].iloc[0]
            evaluation = get_evaluation(mean_vals_dict, label, df_nace_codes_descriptions)
        except IndexError: 
            label_description = ""
            evaluation = {}

        recording.append({"name": os.path.basename(report_path), "NACE": label, "label_description": label_description, **evaluation})

        df_recording = pd.DataFrame(recording)
        df_recording.to_csv(os.path.join(result_path, "recordings.csv"))

        del df_short
        del df_similarities

    return df_recording



def test_report_classification(reports_path: List[str], preprocess_report: callable, threshold_min_chunk_len, cos_threshold, report_to_nace_class, result_path, level=1, overwrite=True): 

    recording = [] 

    df_nace_codes_descriptions = pd.read_csv("../data/NACE_Rev2_Structure_Explanatory_Notes_EN__1_.tsv", sep="\t")

    for report_path in tqdm.tqdm((reports_path)): 

        label = report_to_nace_class.get(os.path.basename(report_path))
        level_1_label = get_all_level(label)[1]

        #print("Report: ", report_path)

        # create report folder 
        store_result_at = os.path.join(result_path, os.path.basename(report_path))
        os.makedirs(store_result_at, exist_ok=True)

        if overwrite: 
            if os.path.exists(store_result_at + "/" + os.path.basename(report_path) + "_long.csv"):
                continue

        # retrieve chunks
        chunks = preprocess_report(pdf_path=report_path)

        # remove chunks with length smaller than threshold
        chunks = [chunk for chunk in chunks if len(chunk) > threshold_min_chunk_len]
        
        # remove duplicates
        chunks = list(set(chunks))
        
        if len(chunks) == 0:
            continue

        #print("Number of Chunks: ", len(chunks))

        # get similarities
        df_similarities = create_sentence_nace_code_similarities.create_sentence_nace_code_similarities(chunks, level=level, df_nace_codes_descriptions=df_nace_codes_descriptions)

        # # plot 
        # fig1 = analysis_functions.plot_mean_scores(df_similarities, cos_threshold=cos_threshold, NACE_code=report_to_nace_class.get(os.path.basename(report_path)), name=os.path.basename(report_path))
        # fig2 = analysis_functions.plot_similarity_distributions(df_similarities, cos_threshold=cos_threshold, NACE_code=report_to_nace_class.get(os.path.basename(report_path)), name=os.path.basename(report_path))
        # fig3 = analysis_functions.plot_nbr_threshold(df_similarities, cos_threshold=cos_threshold, NACE_code=report_to_nace_class.get(os.path.basename(report_path)), name=os.path.basename(report_path))    

        # # store the figures
        # fig1.savefig(os.path.join(result_path, os.path.basename(report_path), "mean_scores.png"), bbox_inches="tight")
        # fig2.savefig(os.path.join(result_path, os.path.basename(report_path), "similarity_distributions.png"), bbox_inches="tight")
        # fig3.savefig(os.path.join(result_path, os.path.basename(report_path), "nbr_threshold.png"), bbox_inches="tight")

        scores_column_names = [column for column in df_similarities.columns if "Scores" in column]

        # apply threshold on similarities
        df_temp = df_similarities[scores_column_names][df_similarities[scores_column_names] > cos_threshold]   

        # replace na vals with 0
        df_temp = df_temp.fillna(0)

        # get mean values for each nace code
        mean_vals = df_temp.mean().sort_values(ascending=False)

        # make folder 
        store_sentences_path = os.path.join(result_path, os.path.basename(report_path), "relevant_sentences_" + os.path.basename(report_path))
        os.makedirs(store_sentences_path, exist_ok=True)

        # store the 100 most important chunks of the 5 most relevant sectors (shown with mean)
        i = 1
        for sector in df_temp.mean().sort_values(ascending=False)[:5].index: 
            top_chunks = df_similarities[sector].sort_values(ascending=False)[:100]
            top_chunks_text = "\n\n".join([f"Score {round(df_similarities.loc[idx][sector], 3)}\n"+ df_similarities.loc[idx]["Sentences"] for idx in top_chunks.index if df_similarities.loc[idx][sector] > cos_threshold])
            top_chunks_text = sector + "\n\n\n" + top_chunks_text

            with open(os.path.join(store_sentences_path, str(i) + "_" + sector + ".txt"), "w") as f:
                f.write(top_chunks_text) 
            i += 1

        # store df
        df_similarities = df_similarities.drop(columns=["Embeddings"])
        df_similarities.to_csv(os.path.join(result_path, os.path.basename(report_path), os.path.basename(report_path) + "_long.csv"))
        df_short = shorten_csv(df_similarities)
        df_short.to_csv(os.path.join(result_path, os.path.basename(report_path), os.path.basename(report_path) + "_short.csv"))

        # record the results 
        mean_vals_dict = {k[7:]:round(v,3) for k,v in mean_vals.to_dict().items()}
        
        # get label of the report
        label = report_to_nace_class.get(os.path.basename(report_path))
        
        try: 
            if isinstance(label, float) or isinstance(label, int):                     
                if len(str(label).split(".")[0]) == 1: 
                    label = "0" + str(label) 
            label_description = df_nace_codes_descriptions[df_nace_codes_descriptions["CODE"] == str(label)]["NAME"].iloc[0]
            evaluation = get_evaluation(mean_vals_dict, label, df_nace_codes_descriptions)
        except IndexError: 
            label_description = ""
            evaluation = {}

        recording.append({"name": os.path.basename(report_path), "NACE": label, "label_description": label_description, **evaluation})

        df_recording = pd.DataFrame(recording)
        df_recording.to_csv(os.path.join(result_path, "recordings.csv"))

        del df_short
        del df_similarities

    return df_recording

