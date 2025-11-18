import os
import torch
import numpy as np
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification
from safetensors.torch import load_file  # comes with HF if safetensors installed
import tqdm
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


def load_custom_bert_from_checkpoint(ckpt_path: str):
    # 1. Load the saved config (includes custom_hidden, custom_num_layers)
    config = AutoConfig.from_pretrained(ckpt_path)

    # 2. Build a model from config (bare BertForSequenceClassification)
    model = AutoModelForSequenceClassification.from_config(config)

    # 3. Rebuild the SAME classifier architecture as in training
    hidden = getattr(config, "custom_hidden", 512)  # fallback if not in config
    num_layers = getattr(config, "custom_num_layers", 2)

    layers = []
    for i in range(num_layers):
        in_dim = config.hidden_size if i == 0 else hidden
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(0.2))

    layers.append(nn.Linear(hidden, config.num_labels))
    model.classifier = nn.Sequential(*layers)

    # 4. Load weights from model.safetensors
    state_dict = load_file(os.path.join(ckpt_path, "model.safetensors"))
    model.load_state_dict(state_dict, strict=True)  # will fail loudly if mismatch

    # 5. Inference mode
    model.eval()
    return model


def BERT_classification_chunk(chunk: str, model, tokenizer):

    inputs = tokenizer(chunk, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(0)
        label_scores = {
            model.config.id2label[i]: logits[i].item()
            for i in range(logits.size(0))
        }

    return label_scores


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

    for chunk in tqdm.tqdm(chunks):

        label_scores = BERT_classification_chunk(chunk, 
                                           model=model, 
                                           tokenizer=tokenizer)
        chunk_logits.append(list(label_scores.values()))
        
        # label_scores = {
        #     model.config.id2label[i]: logits[i].item()
        #     for i in range(logits.size(0))
        # }
        sentence_classification.append(label_scores)

    if len(chunk_logits) == 0:
        return {}

    # 3) Aggregate logits over all chunks (mean over chunks)
    stacked = np.array(chunk_logits)  # (num_chunks, num_labels)
    avg_logits = np.mean(stacked, 0)            # (num_labels,)

    # Map to human-readable labels
    label_scores = {
        model.config.id2label[i]: avg_logits[i].item()
        for i in range(avg_logits.size(0))
    }

    sentence_classification = pd.DataFrame(sentence_classification)
    sentence_classification["Sentences"] = chunks
    sentence_classification = sentence_classification[["Sentences"] + [c for c in sentence_classification.columns if c != "Sentences"]]

    os.makedirs(os.path.join(result_path, os.path.basename(report_path)), exist_ok=True)
    sentence_classification.to_csv(os.path.join(result_path, os.path.basename(report_path), os.path.basename(report_path) + "_classifications.csv"))

    return label_scores, sentence_classification