import re
from typing import List
from sentence_splitter import split_text_into_sentences
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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

def preprocess_report(pdf_path: str, sentence_length: int = 6, add_tables: bool = True, **kwargs) -> List[str]:

    with open(pdf_path, "r") as f: 
        text = f.read()
    
    lines = text.split("\n")

    if add_tables:
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

    if add_tables:
        accepted_lines += tables

    return chunks


from typing import List
import re

def preprocess_report_into_bert_chunks(
    pdf_path: str,
    tokenizer,
    max_length: int = None,
    **kwargs,
) -> List[str]:
    """
    Pack sentences into chunks such that each chunk fits into the BERT context
    window (max_length) according to the tokenizer.

    - sentences: list of already-cleaned sentences (strings)
    - tokenizer: a HuggingFace tokenizer (e.g. BertTokenizerFast)
    - max_length: maximum sequence length INCLUDING special tokens

    Returns: list of text chunks (strings)
    """

    if max_length is None: 
        max_length = tokenizer.model_max_length
    
    with open(pdf_path, "r") as f: 
        text = f.read()
    
    sentences = split_text_into_sentences(text, language='en')

    sentences = [re.sub(r'\b\d+\.\d+\b', '', sentence) for sentence in sentences]
    sentences = [re.sub(r"[^a-zA-ZäöüÄÖÜß.\s]", '', sentence) for sentence in sentences]
    sentences = [re.sub(r"\s+", " ", sentence) for sentence in sentences]
    sentences = [re.sub(r'\.{2,}', " ", sentence) for sentence in sentences]
    sentences = [re.sub(r'^\d+\.\s*', " ", sentence) for sentence in sentences]
    sentences = [sentence.lower() for sentence in sentences]
    sentences = [sentence.strip() for sentence in sentences]

    chunks = []
    current_sentences: List[str] = []

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # 1) Handle the case where a single sentence is longer than max_length.
        enc_sent = tokenizer(
            sent,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        if len(enc_sent["input_ids"]) > max_length:
            # Flush current chunk (if any) before splitting long sentence
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []

            # Split the overly long sentence into token windows (max_length - 2 for [CLS], [SEP])
            token_ids = tokenizer.encode(sent, add_special_tokens=False)
            window_size = max_length - 2  # leave room for special tokens
            for i in range(0, len(token_ids), window_size):
                window_ids = token_ids[i:i + window_size]
                chunk_text = tokenizer.decode(window_ids, skip_special_tokens=True)
                chunks.append(chunk_text.strip())
            continue

        # 2) Normal case: sentence fits individually, try to add it to current chunk.
        if not current_sentences:
            # Start a new chunk
            current_sentences.append(sent)
        else:
            tentative_text = " ".join(current_sentences + [sent])
            enc_tentative = tokenizer(
                tentative_text,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )

            if len(enc_tentative["input_ids"]) <= max_length:
                # It still fits, keep growing current chunk
                current_sentences.append(sent)
            else:
                # Close current chunk and start a new one with this sentence
                chunks.append(" ".join(current_sentences))
                current_sentences = [sent]

    # Flush last chunk
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks

if __name__ == "__main__": 

    ckpt_path = "results/BERT_models/NACE_classification/037_results__data_approach_3__desc_lvl_level_1_dataset_2__num_layers_1__cos_thres_0.5bert-base-uncased__train_full_model__some_labels__only_labels/checkpoint-743"
    ckpt_path = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path) 

    #print(preprocess_report('/Users/hendrikweichel/Downloads/S.S. Lazio S.p.A.3.txt'))
    #print(preprocess_report_into_bert_chunks('/Users/hendrikweichel/Downloads/S.S. Lazio S.p.A.3.txt', tokenizer))
    #print(preprocess_report('/Users/hendrikweichel/Downloads/S.S. Lazio S.p.A.3.txt'))
    chunks = preprocess_report_into_bert_chunks('data/datasets/stoxx_600/TXTs/ABB Ltd.2.txt', tokenizer)
    for chunk in chunks[:10f]: 
        print()
        print()
        print(chunk)
        print(tokenizer(chunk, return_tensors="pt", truncation=True)["input_ids"].shape[1])