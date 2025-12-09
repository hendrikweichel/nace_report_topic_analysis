import numpy as np
import pandas as pd
import os
import sys
import glob
import tqdm
import re
from typing import List

current_dir = "/data/resources/weichel-llama3/work/projects/nace_classification/nace_report_topic_analysis"

sys.path.append("..")
sys.path.append(os.path.join(current_dir, "test"))
sys.path.append(current_dir)

from src import text_extraction, create_sentence_nace_code_similarities, analysis_functions
import test_base
from preprocessing import preprocess_report
from sentence_splitter import split_text_into_sentences

# Parameters: 

threshold_min_chunk_len = 100
cos_threshold = 0.4
sentence_length = 6

# Parameters: 

threshold_min_chunk_len = 100
cos_threshold = 0.4
sentence_length = 6

#dataset_path = "../data/german_annual_reports"
#dataset_path = "../data/stoxx_600_extended"
#dataset_path = "../data/stoxx_600"
dataset_path = os.path.join(current_dir, "data/reports_subset_from_full_data_1")

over_view_df_path = os.path.join(dataset_path, os.path.basename(dataset_path) + "_overview.csv")

print(current_dir)
print(dataset_path)
print(over_view_df_path)

dataset_path_texts = os.path.join(dataset_path, "TXTs")

dataset_name = os.path.basename(dataset_path)

nace_classes = pd.read_csv(over_view_df_path, index_col=0)


report_to_nace_class = nace_classes.dropna(subset=["Report"]).set_index('Report').to_dict()["NACE"]

report_to_nace_class = {report[0][:-4] + ".txt": report[1] for report in report_to_nace_class.items()}
reports_path = glob.glob(os.path.join(dataset_path_texts, "*.txt"))

df_nace_codes_descriptions = pd.read_csv(os.path.join(current_dir, "data/NACE_Rev2_Structure_Explanatory_Notes_EN__1_.tsv"), sep="\t")
    
for i in range(1, 2): 
    nace_level = i

    result_path = f"../results/dataset_{dataset_name}__sentence_len_{sentence_length}__min_chunk_len_{threshold_min_chunk_len}__cos_thresh_{cos_threshold}__nace_level_{nace_level}"

    res = test_base.test_similarities(reports_path, preprocess_report, threshold_min_chunk_len, cos_threshold, report_to_nace_class, result_path, level=i, df_nace_codes_descriptions=df_nace_codes_descriptions)