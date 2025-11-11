import pandas as pd
import numpy as np
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from functools import lru_cache

### Helper

alphabet = "abcdefghijklmnopqrstuvwxyz"


from transformers import AutoTokenizer
import numpy as np

def _chunk_by_tokens(text: str, tokenizer, max_tokens: int, overlap: int = 20):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return []
    step = max_tokens - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than max_tokens")

    chunks = []
    for start in range(0, len(ids), step):
        piece = ids[start:start+max_tokens]
        chunks.append(tokenizer.decode(piece, skip_special_tokens=True))
    return chunks

def embed_document_no_trunc(
    embedding_model,
    texts,
    overlap: int = 20,
    safety_margin: int = 2,
    batch_size: int = 64,
    pooling: str = "mean",          # "mean" or "length_weighted"
    renormalize: bool = True        # re-normalize per-doc after pooling
):
    """
    Returns:
        doc_embeddings: np.ndarray of shape (len(texts), dim)
    """
    model_name = embedding_model.model_name
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Get a sane token budget
    max_len = int(getattr(tok, "model_max_length", 512))
    if max_len > 10000:  # some tokenizers report huge sentinel values
        max_len = 512
    budget = max(8, max_len - safety_margin)

    # Build chunks and bookkeeping
    all_chunks, doc_ranges, chunk_lengths = [], [], []
    start = 0
    for text in texts:
        chunks = _chunk_by_tokens(text or "", tok, budget, overlap=overlap)
        if not chunks:  # empty text → create an empty placeholder chunk
            chunks = [""]
        all_chunks.extend(chunks)
        # keep where each doc's chunks live in all_chunks
        end = start + len(chunks)
        doc_ranges.append((start, end))
        # measure chunk lengths (in tokens) for weighted pooling
        chunk_lengths.extend([len(tok.encode(c, add_special_tokens=False)) for c in chunks])
        start = end

    # Embed in batches
    all_vecs = []
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        all_vecs.extend(embedding_model.embed_documents(batch))
    all_vecs = np.asarray(all_vecs, dtype=float)

    # Pool back to one vector per original text
    dim = all_vecs.shape[1]
    doc_embeddings = np.zeros((len(texts), dim), dtype=float)

    for di, (lo, hi) in enumerate(doc_ranges):
        V = all_vecs[lo:hi]
        if pooling == "length_weighted":
            w = np.array(chunk_lengths[lo:hi], dtype=float)
            w = w / (w.sum() + 1e-12)
            pooled = (w[:, None] * V).sum(axis=0)
        else:  # "mean"
            pooled = V.mean(axis=0)

        if renormalize:
            norm = np.linalg.norm(pooled) + 1e-12
            pooled = pooled / norm
        doc_embeddings[di] = pooled

    return doc_embeddings



@lru_cache(maxsize=1)
def get_embedder():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={
            "device": device,            
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 64,        
        },
    )
    return embedding_model

def cosine_similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))


def token_split_tokens(text, max_len, model_name, overlap=40):
    
    tok = AutoTokenizer.from_pretrained(model_name)
    tks = tok.tokenize(text, add_special_tokens=False)
    for i in range(0, len(tks), max_len - overlap - 2):  # reserve 2 spots
        sub = tks[i : i + max_len - 2]
        ids = [tok.cls_token_id] + tok.convert_tokens_to_ids(sub) + [tok.sep_token_id]
        yield tok.decode(ids)            # or return the ids directly

def embed_document(embedding_model, texts: list): 
    embeddings = []
    tok = AutoTokenizer.from_pretrained(embedding_model.model_name)
    for text in texts: 
        chunks = list(token_split_tokens(text, tok.model_max_length, embedding_model.model_name))
        vectors = np.array(embedding_model.embed_documents(chunks))
        embeddings.append(np.array([vectors.mean(axis=0)])[0])
    return np.array(embeddings)

def get_labels(df_nace_codes_descriptions, level): 
    
    df_descriptions = df_nace_codes_descriptions[df_nace_codes_descriptions["LEVEL"]==level].copy()
    
    if level == 3: 
        for i, row in df_descriptions.iterrows(): 
        
            # if there is no discription of the NACE code
            if not pd.notna(row["Includes"]): 

                # Try to get the title of the nace code with lower level and same name
                new_row = df_nace_codes_descriptions[(df_nace_codes_descriptions["CODE"] == row["CODE"] + "0") & (df_nace_codes_descriptions["NAME"] == row["NAME"])]
        
                if len(new_row) == 1:
                    if not pd.isna(new_row["Includes"].iloc[0]): 
                        df_descriptions.loc[i, "Includes"] = new_row["Includes"].iloc[0]
                        df_descriptions.loc[i, "IncludesAlso"] = new_row["IncludesAlso"].iloc[0]
                        continue
                    
                        
                # If this didn't work, concatenate the lower levels
                children = df_nace_codes_descriptions[df_nace_codes_descriptions["PARENT_ID"] == row["ID"]]
                new_includes = ". ".join(children["Includes"].dropna())
                new_includes_also = ". ".join(children["IncludesAlso"].dropna())
                df_descriptions.loc[i, "Includes"] = new_includes
                df_descriptions.loc[i, "AlsoIncludes"] = new_includes_also

                if df_descriptions.loc[i, "Includes"] == "":
                    df_descriptions.loc[i, "Includes"] = new_row["NAME"].iloc[0]

    df_descriptions.loc[:,"Includes"] = df_descriptions["Includes"].fillna("")
    df_descriptions.loc[:,"IncludesAlso"] = df_descriptions["IncludesAlso"].fillna("")
    labels = [df_descriptions.loc[:,"Includes"].iloc[i] + " " + df_descriptions.loc[:, "IncludesAlso"].iloc[i] for i in range(len(df_descriptions))]
    df_descriptions.loc[:,"Labels"] = labels
    return df_descriptions

@lru_cache()
def embedd_docs(chunks, embedding_model): 
    return embedding_model.embed_documents(chunks)

def create_sentence_nace_code_similarities(chunks: str, level = 1, df_nace_codes_descriptions = None) -> pd.DataFrame:
    
    if df_nace_codes_descriptions is None: 
        df_nace_codes_descriptions = pd.read_csv("../data/NACE_Rev2_Structure_Explanatory_Notes_EN__1_.tsv", sep="\t")

    # preprocess the description of the text NACE Codes 
    df_descriptions = get_labels(df_nace_codes_descriptions, level)

    embedding_model = get_embedder()
    
    ### Embed sentences
    df_sentences = pd.DataFrame(chunks, columns=["Sentences"])
    df_sentences["Embeddings"] = embedding_model.embed_documents(df_sentences["Sentences"].to_list())
    #df_sentences["Embeddings"] = embedd_docs(df_sentences["Sentences"].to_list(), embedding_model)

    # Embed classes
    df_descriptions["Embeddings"] = embedding_model.embed_documents(df_descriptions["Labels"].to_list())
    #df_descriptions["Embeddings"] = embedd_docs(df_sentences["Labels"].to_list(), embedding_model)

    # Caclulate Similarities
    dict_similarities = {}
    for i, row in df_descriptions.iterrows(): 
        similarities = df_sentences["Embeddings"].apply(lambda x: cosine_similarity(x, row.Embeddings))
        dict_similarities[f"Scores_{row.CODE}_{row.NAME}"] = similarities
    
    df_sentences = pd.concat([df_sentences, pd.DataFrame(dict_similarities, dtype=float, index=df_sentences.index)], axis=1)

    return df_sentences