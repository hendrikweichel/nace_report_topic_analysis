import re
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import glob
import os

# --------------------------
# 1. Simple sentence splitter
# --------------------------

def split_into_sentences(text: str) -> List[str]:
    """
    Very simple sentence splitter.
    Splits on '.', '!' or '?' followed by whitespace.
    Not perfect, but good enough to illustrate the idea.
    """
    text = text.strip()
    if not text:
        return []

    # Split on end-of-sentence punctuation
    parts = re.split(r'(?<=[.!?])\s+', text)
    # Remove empty pieces
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences

# -------------------------------------------
# 2. Build TF-IDF model on training sentences
# -------------------------------------------
def build_tfidf_model(training_texts: List[str]) -> TfidfVectorizer:
    """
    training_texts: list of full report texts (strings)
    Returns a fitted TfidfVectorizer over all sentences in the training corpus.
    """

    # Initialize the TF-IDF vectorizer.
    # You can remove 'stop_words' if you don't want English stopword removal.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english"  # optional
    )

    vectorizer.fit(training_texts)
    return vectorizer

# ------------------------------------------------
# 3. Rank sentences in a single report using TF-IDF
# ------------------------------------------------
def rank_sentences(report_paragraphs: list, vectorizer: TfidfVectorizer) -> List[tuple]:
    """
    report_text: the text of a single report
    vectorizer: a fitted TfidfVectorizer
    Returns a list of (sentence, score, original_index) sorted by score (desc).
    """

    # Transform sentences into TF-IDF matrix: shape (num_sentences, vocab_size)
    tfidf_matrix = vectorizer.transform(report_paragraphs)

    # Simple sentence score: sum of TF-IDF values of all terms in that sentence
    # shape: (num_sentences, 1) -> flatten to (num_sentences,)
    scores = np.asarray(tfidf_matrix.sum(axis=1)).ravel()

    # Create list of (sentence, score, original_index)
    sentence_info = [
        (report_paragraphs[i], float(scores[i]), i) for i in range(len(report_paragraphs))
    ]

    # Sort by score descending
    sentence_info_sorted = sorted(sentence_info, key=lambda x: x[1], reverse=True)
    return sentence_info_sorted

# ----------------------------------------------------------------------
# 4. Select top sentences under a word budget and restore original order
# ----------------------------------------------------------------------
def select_top_sentences(
    report_paragraphs: list,
    vectorizer: TfidfVectorizer,
    max_words: int = 512
) -> str:
    """
    report_text: text of a single report
    vectorizer: fitted TfidfVectorizer
    max_words: approximate word budget (e.g. ~512 words ~= 512-800 tokens)

    Returns a shortened text containing only the selected sentences
    in their ORIGINAL order.
    """
    # Rank sentences by importance
    ranked = rank_sentences(report_paragraphs, vectorizer)
    if not ranked:
        return ""

    # Greedy selection: go down ranked list, keep sentence if it fits in budget
    selected_indices = []
    word_count = 0

    for sentence, score, idx in ranked:
        num_words = len(sentence.split())
        if word_count + num_words <= max_words:
            selected_indices.append(idx)
            word_count += num_words

    # Restore original sentence order
    selected_indices_sorted = sorted(selected_indices)

    # Build final shortened text
    selected_sentences = [report_paragraphs[i] for i in selected_indices_sorted]
    shortened_text = " ".join(selected_sentences)

    return selected_sentences, ranked

if __name__ == "__main__": 

    train_df = pd.read_csv("projects/nace_classification/nace_report_topic_analysis/data/training_data/dataset_reports_subset_from_full_data_1__sentence_len_6__min_chunk_len_100__cos_thresh_0.4__nace_level_1__2nd_approach__nace_level_1__cos_thres_0.4/full_data.csv")
    test_reports = glob.glob("projects/nace_classification/nace_report_topic_analysis/results/BERT_classification/model_1_0__dataset__reports_subset_from_full_data_3_sentence_len_6__nace_level_1/*/*.csv")

    training_reports = train_df["text"].to_list()
    #Build TF-IDF model on all sentences from the training corpus
    tfidf_vectorizer = build_tfidf_model(training_reports)
    # Example new report that we want to shorten
    report = test_reports[3]
    new_report = pd.read_csv(report)["Sentences"].to_list()
    print(os.path.basename(report))

    shortened, ranked = select_top_sentences(
        report_paragraphs=new_report,
        vectorizer=tfidf_vectorizer,
        max_words=2000
    )

    print("=== ORIGINAL REPORT ===")
    print(new_report)
    print("\n=== SHORTENED REPORT ===")
    for s in shortened: 
        print(s) 
    print(ranked)