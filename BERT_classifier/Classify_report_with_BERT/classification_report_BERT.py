from scipy.special import softmax
import os
import torch
import numpy as np
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification
from safetensors.torch import load_file  # comes with HF if safetensors installed
import tqdm
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import os
from cachetools import LFUCache

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


def load_custom_bert_from_checkpoint(ckpt_path: str, num_layers_base: int = 2):
    # 1. Load the saved config (includes custom_hidden, custom_num_layers)
    config = AutoConfig.from_pretrained(ckpt_path)

    # 2. Build a model from config (bare BertForSequenceClassification)
    model = AutoModelForSequenceClassification.from_config(config)

    # 3. Rebuild the SAME classifier architecture as in training
    hidden = getattr(config, "custom_hidden", 512)  # fallback if not in config
    num_layers = getattr(config, "custom_num_layers", num_layers_base)

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

    return logits

@LFUCache(1)
def get_relevancy_model(ckpt_path: str = "results/BERT_models/relevancy_judge__2__num_layers_1__cos_thres_0.35__train_full_model_Truebert-base-uncased__train_full_model__all_labels/checkpoint-88"): 
    return load_custom_bert_from_checkpoint(ckpt_path, num_layers_base = 1)

@LFUCache(1)
def get_relevancy_tokenizer(ckpt_path: str = "results/BERT_models/relevancy_judge__2__num_layers_1__cos_thres_0.35__train_full_model_Truebert-base-uncased__train_full_model__all_labels/checkpoint-88"): 
    return AutoTokenizer.from_pretrained(ckpt_path) 

def relevancy_classification(chunk): 
    """ Uses a binary model to classify whether a chunk is relevant or not.

    Args:
        chunk (_type_): _description_

    Returns:
        _type_: _description_
    """

    tik = time.time()
    model_binary = get_relevancy_model()
    tokenizer = get_relevancy_tokenizer()
    print(time.time() - tik)
    inputs = tokenizer(chunk, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model_binary(**inputs)
        logits = outputs.logits.squeeze(0)
        probs = torch.sigmoid(logits) 
        
    return probs.item()

def classify_report(chunks: list, 
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

    for chunk in (chunks):

        logits = BERT_classification_chunk(chunk, 
                                           model=model, 
                                           tokenizer=tokenizer)
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

    # add the relevancy check

    if True: 
        sentence_classification["Relevancy_sig"] = sentence_classification["Sentences"].apply(relevancy_classification)


    os.makedirs(os.path.join(result_path, os.path.basename(report_path)), exist_ok=True)
    sentence_classification.to_csv(os.path.join(result_path, os.path.basename(report_path), os.path.basename(report_path) + "_classifications.csv"))

    return label_scores#, sentence_classification

if __name__ == "__main__": 

    ckpt_path = "results/BERT_models/results__new_approach_data__num_layers_2bert-base-uncased__train_full_model__some_labels/checkpoint-15990"
    model = load_custom_bert_from_checkpoint(ckpt_path)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path) 

    chunk = "any reasonably possible change in the foreign currency exchange rates other than disclosed above at the end of the reporting period against the respective functional currency of the entities within the group does not have a material impact on the profitloss after tax and other comprehensive incomeexpenses"
    chunk = "This section includes the exploitation of vegetal and animal natural resources, comprising the activities of growing of crops, raising and breeding of animals, harvesting of timber and other plants, animals or animal products from a farm or their natural habitats."
    chunk = """Ein Märchen ist eine alte Erzählung, die oft mit "Es war einmal..." beginnt und sich durch Merkmale wie magische Elemente, stereotype Figuren und eine oft unbestimmte Zeit und Ort auszeichnet. Bekannte Beispiele sind die Sammlung der Brüder Grimm (z.B. Hänsel und Gretel, Rotkäppchen, Aschenputtel), obwohl Märchen ursprünglich auch für Erwachsene gedacht waren und als Ratgeber für grundlegende menschliche Erfahrungen dienen können"""

    label_scores = BERT_classification_chunk(chunk, model, tokenizer)
    print(chunk)
    print(softmax(label_scores))
    label_scores = {
        model.config.id2label[i]: label_scores[i].item()
        for i in range(len(label_scores))
    }

    print(label_scores)
    
    #classify_report(chunks=[chunk], model=model, tokenizer=tokenizer, result_path="../../results/BERT_classification/test_1", report_path="data/datasets/stoxx_600/TXTs/Hannover Rueck SE1.txt")
