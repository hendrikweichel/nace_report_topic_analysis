#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import sys
import glob
import tqdm
import re
from typing import List

sys.path.append("..")
sys.path.append("/data/resources/weichel-llama3/work/projects/nace_classification/nace_report_topic_analysis/")
from src import create_sentence_nace_code_similarities, analysis_functions
import test_base
from sentence_splitter import split_text_into_sentences


# ## Test retrieving the similarities for chunks in a pdf to the NACE Code
# 
# **Function:** pdf-> (chunk x code -> [-1,1])
# 
# **Parameters:** 
# 
# - pdf_path
# - way of chunking the text (e.g. sentences, sliding window, or paragraphs)
# - way of preprocessing (most is fixed for all reports)
#     - similarity threshold of relevant chunks
#     - length of irrelevant chunks
# 
# **Store analytics for each datapoint:**
# 
# - mean score for each class given a threshold

# In[2]:
print("Hello, los gehts")


# Parameters: 

threshold_min_chunk_len = 100
cos_threshold = 0.4
sentence_length = 6


# In[3]:

mypath = "/data/resources/weichel-llama3/work/projects/nace_classification/nace_report_topic_analysis/"

os.chdir(mypath)


dataset_path = "../data/datasets/german_annual_reports"
dataset_path = "../data/datasets/stoxx_600"
dataset_path = "../data/datasets/stoxx_600_extended"
dataset_path = "../data/datasets/reports_subset_from_full_data_1"
dataset_path = mypath+"data/datasets/reports_subset_from_full_data_3"
dataset_path = mypath+"data/datasets/reports_subset_from_full_data_1"
dataset_path = mypath+"data/datasets/reports_subset_from_full_data_2"


# In[4]:


over_view_df_path = os.path.join(dataset_path, os.path.basename(dataset_path) + "_overview.csv")

dataset_path_texts = os.path.join(dataset_path, "TXTs")

dataset_name = os.path.basename(dataset_path)


# In[6]:


nace_classes = pd.read_csv(over_view_df_path, index_col=0)
nace_classes.head()


# In[7]:


report_to_nace_class = nace_classes.dropna(subset=["Report"]).set_index('Report').to_dict()["NACE"]
report_to_nace_class


# In[8]:


report_to_nace_class = {report[0][:-4] + ".txt": report[1] for report in report_to_nace_class.items()}
report_to_nace_class


# In[9]:


reports_path = glob.glob(os.path.join(dataset_path_texts, "*.txt"))
reports_path


# In[10]:


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

    if len(chunks) == 0:
        return []
    
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


# In[ ]:

print("Hello", len(reports_path))

for i in range(3,4): 
    nace_level = i

    result_path = mypath+ f"results/dataset__{dataset_name}_sentence_len_{sentence_length}__min_chunk_len_{threshold_min_chunk_len}__cos_thresh_{cos_threshold}__nace_level_{nace_level}"
    #print("Store at: ", result_path)

    res = test_base.test_report_classification(
        reports_path=reports_path, 
        preprocess_report=preprocess_report, 
        report_to_nace_class=report_to_nace_class, 
        result_path = result_path, 
        threshold_min_chunk_len=threshold_min_chunk_len, 
        cos_threshold=cos_threshold,  
        level=i, 
        overwrite=True, 
        classification_function=create_sentence_nace_code_similarities.classification_by_similarities)