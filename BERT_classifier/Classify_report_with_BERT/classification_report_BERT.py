import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import os

def load_model(ckpt_path: str):
    """
    Loads a fine-tuned NACE BERT classifier and tokenizer from a checkpoint path.
    Automatically places model on GPU if available.
    
    Returns:
        model      -- the loaded classifier (in eval mode)
        tokenizer  -- matching tokenizer
        device     -- torch.device("cuda") or torch.device("cpu")
    """
    # Load
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, device


def classification_report_BERT(chunks: list, 
                    model: AutoModelForSequenceClassification,
                    tokenizer: AutoTokenizer,
                    result_path: str,
                    report_path: str, 
                    level: int = None,
                    **kwags) -> dict:
    """
    1. Preprocess a report into text chunks.
    2. Classify each chunk with the BERT model.
    3. Aggregate logits over all chunks and return {label: logit_score}.
    """

    # 1) Preprocess report -> list of chunks
    chunks = [chunk for chunk in chunks if chunk.strip()]

    # Safety: no chunks -> empty result
    if len(chunks) == 0:
        return {}

    model.eval()
    chunk_logits = []   
    sentence_classification = []

    # 2) Classify each chunk
    with torch.no_grad():
        for chunk in chunks:

            inputs = tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            outputs = model(**inputs)
            # shape: (1, num_labels) -> (num_labels,)
            logits = outputs.logits.squeeze(0)
            chunk_logits.append(logits)

            label_scores = {
                model.config.id2label[i]: logits[i].item()
                for i in range(logits.size(0))
            }
            sentence_classification.append(label_scores)

    if len(chunk_logits) == 0:
        return {}
    
    # 3) Aggregate logits over all chunks (mean over chunks)
    stacked = torch.stack(chunk_logits, dim=0)  # (num_chunks, num_labels)
    avg_logits = stacked.mean(dim=0)            # (num_labels,)

    # Map to human-readable labels
    label_scores = {
        model.config.id2label[i]: avg_logits[i].item()
        for i in range(avg_logits.size(0))
    }

    sentence_classification = pd.DataFrame(sentence_classification)
    sentence_classification["Sentences"] = chunks

    os.makedirs(os.path.join(result_path, os.path.basename(report_path)), exist_ok=True)
    sentence_classification.to_csv(os.path.join(result_path, os.path.basename(report_path), os.path.basename(report_path) + "_classifications.csv"))

    #return pd.DataFrame.from_dict(label_scores, orient="index")[0]
    return label_scores



