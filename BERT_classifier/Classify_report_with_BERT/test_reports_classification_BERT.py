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
from preprocessing import preprocess_report


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
dataset_path = wor_dir + "/data/datasets/reports_subset_from_full_data_3"

# In[5]:


over_view_df_path = os.path.join(dataset_path, os.path.basename(dataset_path) + "_overview.csv")

dataset_path_texts = os.path.join(dataset_path, "TXTs")

dataset_name = os.path.basename(dataset_path)


# In[6]:


try:
    nace_classes = pd.read_csv(over_view_df_path, index_col=0, sep=",")
    nace_classes.head()
except pd.errors.ParserError:
    nace_classes = pd.read_csv(over_view_df_path, index_col=0, sep=";")
    
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


# In[11]:
model_version = "1_0"
model_version = "2_1"
model_version = "2_2"
if model_version == "1_0":
    ckpt = wor_dir + "/results/BERT_models/results_null_classifiers__cos_thres_0.5__bert-base-uncased__train_full_model__some_labels/checkpoint-2331"
    num_layers = 1
elif model_version == "2_1":
    ckpt = wor_dir + "/results/BERT_models/_best_2nd_results__new_approach_data__num_layers_2__cos_thres_0.5bert-base-uncased__train_full_model__some_labels/checkpoint-4800"
    num_layers = 2
elif model_version == "2_2":
    ckpt = wor_dir + "/results/BERT_models/NACE_classification/037_results__data_approach_3__desc_lvl_level_1_dataset_2__num_layers_1__cos_thres_0.5bert-base-uncased__train_full_model__some_labels__only_labels/checkpoint-743"
    num_layers = 1

model = classification_report_BERT.load_custom_bert_from_checkpoint(ckpt_path=ckpt, num_layers_base=num_layers)
tokenizer = AutoTokenizer.from_pretrained(ckpt)

# In[ ]:

import random

random.shuffle(reports_path)

print("LETS GO")

for i in range(1,2):
    nace_level = i

    result_path = wor_dir + f"/results/BERT_classification/rel_only_model_{model_version}__with_relevancy__dataset__{dataset_name}_sentence_len_{sentence_length}__nace_level_{nace_level}"
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




