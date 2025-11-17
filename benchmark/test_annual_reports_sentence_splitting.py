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

        def preprocess_report(pdf_path: str) -> List[str]:

            with open(pdf_path, "r") as f: 
                text = f.read()
            
            lines = text.split("\n")

            # drop if condidtion is True
            conditions = [
                # filter images
                lambda line: line == '<!-- image -->',
                
                #filter tables 
                lambda line: (line[0] == "|" and line[-1] == "|") if len(line) > 1 else False, 

                # filter headers
                #lambda line: line.strip()[0] == "#" if len(line) > 0 else True,

                # filter sentences
                #lambda line: "." not in line,
                
                # more than 50% is numbers
                #lambda line: sum(ch.isalpha() for ch in line) / len(line) < 0.5,

                # minimum 3 words 
                #lambda line: len(re.sub(r"[^a-zA-ZäöüÄÖÜß\s]", '', line).strip().split(" ")) < 3,

                # Minimum 2 Sentences
                #lambda line: sum([0 if len(sentence.split(" ")) < 3 else 1 for sentence in split_text_into_sentences(line, "en")]) < 2

            ]
            accepted_lines = [line for line in lines if not any(condition(line) for condition in conditions)]

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

            # Clean up chunks: remove numbers, special characters, extra spaces, and normalize case
            chunks = [re.sub(r'\b\d+\.\d+\b', '', chunk) for chunk in chunks]  # Remove decimal numbers
            chunks = [re.sub(r"[^a-zA-ZäöüÄÖÜß.\s]", '', chunk) for chunk in chunks]  # Remove non-letter characters except dots and spaces
            chunks = [re.sub(r"\s+", " ", chunk) for chunk in chunks]  # Replace multiple spaces with single space
            chunks = [re.sub(r'\.{2,}', " ", chunk) for chunk in chunks]  # Replace multiple dots with space
            chunks = [re.sub(r'^\d+\.\s*', " ", chunk) for chunk in chunks]  # Remove leading numbers followed by dot
            chunks = [chunk.lower() for chunk in chunks]  # Convert to lowercase
            chunks = [chunk.strip() for chunk in chunks]  # Strip leading/trailing whitespace

            return chunks
        
        result_path = f"../results/paragraph_and_sentence_len_{sentence_length}_min_chunk_len_{threshold_min_chunk_len}_cos_thresh_{cos_threshold}_nace_level_{nace_level}_stoxx"

        res = test_base.test_report_classification(reports_path, preprocess_report, threshold_min_chunk_len, cos_threshold, report_to_nace_class, result_path, level=nace_level)
