#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import os
import time
import pandas as pd
import json
import sys
import glob
import tqdm
import re
from typing import List

wor_dir = "/data/resources/weichel-llama3/work/projects/nace_classification/nace_report_topic_analysis"
sys.path.append(wor_dir)
import test_base
from sentence_splitter import split_text_into_sentences
from BERT_classifier.Classify_report_with_BERT import classification_report_BERT
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# In[3]:


sentence_length = 6


# ## Test using a BERT-Classification Model trained on paragraphs to classify an entire report.
# 
# **Function:** pdf -> NACE Class

# In[4]:


dataset_path = "data/datasets/german_annual_reports"
dataset_path = "data/datasets/stoxx_600_extended"
dataset_path = "data/datasets/reports_subset_from_full_data_1"
dataset_path = wor_dir + "/data/datasets/stoxx_600"
dataset_path = wor_dir + "/data/datasets/reports_subset_from_full_data_1"

# In[5]:


over_view_df_path = os.path.join(dataset_path, os.path.basename(dataset_path) + "_overview.csv")

dataset_path_texts = os.path.join(dataset_path, "TXTs")

dataset_name = os.path.basename(dataset_path)


# In[6]:


nace_classes = pd.read_csv(over_view_df_path, index_col=0, sep=",")
nace_classes.head()


# In[7]:


report_to_nace_class = nace_classes.dropna(subset=["Report"]).set_index('Report').to_dict()["NACE"]
report_to_nace_class = {report[0][:-4] + ".txt": report[1] for report in report_to_nace_class.items()}
len(report_to_nace_class)


# In[8]:


reports_path = glob.glob(os.path.join(dataset_path, "PDFs/*.pdf"))
len(reports_path)


# In[9]:


reports_path = glob.glob(os.path.join(dataset_path_texts, "*.txt"))
len(reports_path)


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


# In[11]:
model_version = "2_1"
model_version = "1_0"
if model_version == "1_0":
    ckpt = wor_dir + "/results/BERT_models/results_null_classifiers__cos_thres_0.5__bert-base-uncased__train_full_model__some_labels/checkpoint-2331"
    num_layers = 1
elif model_version == "2_1":
    ckpt = wor_dir + "/results/BERT_models/_best_2nd_results__new_approach_data__num_layers_2__cos_thres_0.5bert-base-uncased__train_full_model__some_labels/checkpoint-4800"
    num_layers = 2

model = classification_report_BERT.load_custom_bert_from_checkpoint(ckpt_path=ckpt, num_layers_base=num_layers)
tokenizer = AutoTokenizer.from_pretrained(ckpt)

# In[ ]:

import random

random.shuffle(reports_path)

print("LETS GO")

for i in range(1,2):
    nace_level = i

    result_path = wor_dir + f"/results/BERT_classification/model_{model_version}__with_relevancy__dataset__{dataset_name}_sentence_len_{sentence_length}__nace_level_{nace_level}"
    os.makedirs(result_path, exist_ok=True)
    config = {"model": ckpt, "dataset": dataset_name}
    with open(os.path.join(result_path, "config.json"), "w") as f: 
        json.dump(config, f)

    res = test_base.test_report_classification(
        reports_path=reports_path,
        preprocess_report=preprocess_report, 
        report_to_nace_class=report_to_nace_class, 
        result_path = result_path,
        level=i,
        overwrite=True,
        classification_function=classification_report_BERT.classify_report, 
        path_nace_code_descriptions=wor_dir+"/data/NACE_Rev2_Structure_Explanatory_Notes_EN__1_.tsv", 
        model=model,
        tokenizer=tokenizer,)

    

# In[ ]:





# In[ ]:




