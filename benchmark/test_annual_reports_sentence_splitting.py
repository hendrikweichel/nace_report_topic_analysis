import numpy as np
import pandas as pd
import os
import sys
import glob
import tqdm
import re
from typing import List

sys.path.append("..")
from src import text_extraction, create_sentence_nace_code_similarities, analysis_functions
import test_base
from preprocessing import preprocess_report
from sentence_splitter import split_text_into_sentences

# ## Test retrieving the similarities for chunks in a pdf to the NACE Code

# **Function:** pdf-> (chunk x code -> [-1,1])

# **Parameters:** 

# - pdf_path
# - way of chunking the text (e.g. sentences, sliding window, or paragraphs)
# - way of preprocessing (most is fixed for all reports)
#     - similarity threshold of relevant chunks
#     - length of irrelevant chunks

# **Store analytics for each datapoint:**

# - mean score for each class given a threshold
# # Parameters: 

threshold_min_chunk_len = 100
threshold_min_paragraph_len = 0
cos_threshold = 0.4
sentence_length = 3
dataset_path = "../data/german_annual_reports"
dataset_path = "../data/PDF_stoxx600"
dataset_path_texts = "../data/TEXT_stoxx600_docling"
nace_classes = pd.read_excel(os.path.join(dataset_path, "STOXX600_as_of_2025_03_13.xlsx"))
nace_classes.head()

report_to_nace_class = nace_classes.dropna(subset=["Report"]).set_index('Report').to_dict()["NACE"]

report_to_nace_class = {report[0][:-4] + ".txt": report[1] for report in report_to_nace_class.items()}

reports_path = glob.glob(os.path.join(dataset_path_texts, "*.txt"))

for i in [1,2,3,4]:

    sentence_length = 6
    nace_level = i

    for cos_threshold in np.arange(0, 0.6, 0.05):

        result_path = f"../results/paragraph_and_sentence_len_{sentence_length}_min_chunk_len_{threshold_min_chunk_len}_cos_thresh_{cos_threshold}_nace_level_{nace_level}_stoxx"

        res = test_base.test_report_classification(reports_path, preprocess_report, threshold_min_chunk_len, cos_threshold, report_to_nace_class, result_path, level=nace_level)
