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

def get_tables(lines: list): 
    tables = []
    current_table = []

    for line in lines:
        if line.strip().startswith("|"):  # line belongs to a table
            current_table.append(line.strip())
        else:
            if current_table:  # table ended
                tables.append("\n".join(current_table))
                current_table = []

    # catch last table if file ends without empty lines
    if current_table:
        tables.append("\n".join(current_table))

    return tables

def preprocess_report(pdf_path: str) -> List[str]:

    with open(pdf_path, "r") as f: 
        text = f.read()
    
    lines = text.split("\n")

    tables = get_tables(lines)

    # drop if condidtion is True
    conditions = [
        # filter images
        lambda line: line == '<!-- image -->',
        
        #filter tables 
        lambda line: (line[0] == "|" and line[-1] == "|") if len(line) > 1 else False, 

        # filter headers
        lambda line: line.strip()[0] == "#" if len(line) > 0 else True,

        # filter sentences
        lambda line: "." not in line,
        
        # more than 50% is numbers
        lambda line: sum(ch.isalpha() for ch in line) / len(line) < 0.5,

        # minimum 3 words 
        lambda line: len(re.sub(r"[^a-zA-ZäöüÄÖÜß\s]", '', line).strip().split(" ")) < 3,

        # Minimum 2 Sentences
        #lambda line: sum([0 if len(sentence.split(" ")) < 3 else 1 for sentence in split_text_into_sentences(line, "en")]) < 2

    ]
    accepted_lines = [line for line in lines if not any(condition(line) for condition in conditions)]
    accepted_lines += tables

    chunks = []

    for line in accepted_lines: 
        sentences = split_text_into_sentences(line, language='en')
        sentences = [sentence.strip() for sentence in sentences]
        sentences = [sentence for sentence in sentences if sentence != ""]
        new_chunks = [(" ".join(sentences[i:i+sentence_length])).strip() for i in range(0, len(sentences), 3)]

        chunks += new_chunks

    # if there is only one sentence in the last chunk, balance the two last chunks
    if len(split_text_into_sentences(chunks[-1], language = "en")) == 1: 
        last_two_chunks = chunks[-2] + " " + chunks[-1]
        chunks[-2] = last_two_chunks[0:(len(last_two_chunks) + 1) // 2]
        chunks[-1] = last_two_chunks[(len(last_two_chunks) + 1) // 2: (len(last_two_chunks)) - (len(last_two_chunks) + 1) // 2]

    chunks = [re.sub(r'\b\d+\.\d+\b', '', chunk) for chunk in chunks]
    chunks = [re.sub(r"[^a-zA-ZäöüÄÖÜß.\s]", '', chunk) for chunk in chunks]
    chunks = [re.sub(r"\s+", " ", chunk) for chunk in chunks]
    chunks = [re.sub(r'\.{2,}', " ", chunk) for chunk in chunks]
    chunks = [re.sub(r'^\d+\.\s*', " ", chunk) for chunk in chunks]
    chunks = [chunk.lower() for chunk in chunks]
    chunks = [chunk.strip() for chunk in chunks]

    return chunks

df_nace_codes_descriptions = pd.read_csv(os.path.join(current_dir, "data/NACE_Rev2_Structure_Explanatory_Notes_EN__1_.tsv"), sep="\t")
    
for i in range(1, 2): 
    nace_level = i

    result_path = f"../results/dataset_{dataset_name}__sentence_len_{sentence_length}__min_chunk_len_{threshold_min_chunk_len}__cos_thresh_{cos_threshold}__nace_level_{nace_level}"

    res = test_base.test_similarities(reports_path, preprocess_report, threshold_min_chunk_len, cos_threshold, report_to_nace_class, result_path, level=i, df_nace_codes_descriptions=df_nace_codes_descriptions)