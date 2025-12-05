import os
import torch
import numpy as np
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification
from safetensors.torch import load_file  # comes with HF if safetensors installed
import tqdm
import json
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import os
#from cachetools import lru_cache
from functools import lru_cache

wor_dir = "/data/resources/weichel-llama3/work/projects/nace_classification/nace_report_topic_analysis"


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


def load_custom_bert_from_checkpoint(ckpt_path: str, num_layers_base: int = 2, num_labels_base: int = None):
    # 1. Load the saved config (includes custom_hidden, custom_num_layers)
    config = AutoConfig.from_pretrained(ckpt_path)

    # 2. Build a model from config (bare BertForSequenceClassification)
    model = AutoModelForSequenceClassification.from_config(config)

    # 3. Rebuild the SAME classifier architecture as in training
    hidden = getattr(config, "custom_hidden", 512)  # fallback if not in config
    num_layers = getattr(config, "custom_num_layers", num_layers_base)
    num_labels = getattr(config, "num_labels", num_labels_base)

    layers = []
    for i in range(num_layers):
        in_dim = config.hidden_size if i == 0 else hidden
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(0.2))

    layers.append(nn.Linear(hidden, num_labels))
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

@lru_cache(1)
def get_relevancy_model(ckpt_path: str = wor_dir + "/results/BERT_models/relevancy_judge__2__num_layers_1__cos_thres_0.35__train_full_model_Truebert-base-uncased__train_full_model__all_labels/checkpoint-88"): 
    # load checkpoint
    config = AutoConfig.from_pretrained(ckpt_path)
    
    # 2. Build a model from config (bare BertForSequenceClassification)
    model_binary = AutoModelForSequenceClassification.from_config(config)
    
    # 3. Rebuild the SAME classifier architecture as in training
    hidden = getattr(config, "custom_hidden", 512)  # fallback if not in config
    num_layers = getattr(config, "custom_num_layers", 1)
    
    layers = []
    for i in range(num_layers):
        in_dim = config.hidden_size if i == 0 else hidden
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(0.2))
    layers.append(nn.Linear(hidden, 1))
    model_binary.classifier = nn.Sequential(*layers)
    
    # 4. Load weights from model.safetensors
    state_dict = load_file(os.path.join(ckpt_path, "model.safetensors"))
    model_binary.load_state_dict(state_dict, strict=True)  # will fail loudly if mismatch
    
    # 5. Inference mode
    model_binary.eval()
    
    return model_binary

@lru_cache(1)
def get_relevancy_tokenizer(ckpt_path: str = wor_dir + "/results/BERT_models/relevancy_judge__2__num_layers_1__cos_thres_0.35__train_full_model_Truebert-base-uncased__train_full_model__all_labels/checkpoint-88"): 
    return AutoTokenizer.from_pretrained(ckpt_path) 

def relevancy_classification(chunk): 
    """ Uses a binary model to classify whether a chunk is relevant or not.

    Args:
        chunk (_type_): _description_

    Returns:
        _type_: _description_
    """
    #tik = time.time()
    model_binary = get_relevancy_model()
    tokenizer = get_relevancy_tokenizer()
    inputs = tokenizer(chunk, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model_binary(**inputs)
        logits = outputs.logits.squeeze(0)
        probs = torch.sigmoid(logits) 
    #print(time.time()-tik)
    return probs.item()

def softmax(x):
    e = np.exp(x - np.max(x))   # subtract max for numerical stability
    return e / e.sum()

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

    tik = time.time()    
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
    time_BERT_Classification = time.time()-tik
    
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
    scores = sentence_classification.columns.to_list()
    
    sentence_classification.loc[:,"classification"] = [str(row.sort_values(ascending=False).to_dict()) for i, row in sentence_classification[scores].apply(softmax, axis=1).iterrows()]
    sentence_classification.loc[:,"max_class_sim"] = [model.config.id2label[i] for i in np.argmax(sentence_classification[scores], 1)]
    sentence_classification["Sentences"] = chunks

    # add the relevancy check

    tik = time.time()    
    if True: 
        sentence_classification["Relevancy_sig"] = sentence_classification["Sentences"].apply(relevancy_classification)
        sentence_classification["Relevancy_over_0_5"] = sentence_classification["Relevancy_sig"] > 0.5
    time_relevancy = time.time()-tik
    
    os.makedirs(os.path.join(result_path, os.path.basename(report_path)), exist_ok=True)
    sentence_classification.to_csv(os.path.join(result_path, os.path.basename(report_path), os.path.basename(report_path) + "_classifications.csv"))
    with open(os.path.join(result_path, os.path.basename(report_path), os.path.basename(report_path) + "log.json"), "w") as f: 
        timing = {
            "time_relevancy": time_relevancy,
            "time_BERT_Classification": time_BERT_Classification
        }
        json.dump(timing, f)

    return label_scores#, sentence_classification

if __name__ == "__main__": 

    ckpt_path = "results/BERT_models/results__new_approach_data__num_layers_2bert-base-uncased__train_full_model__some_labels/checkpoint-15990"
    model = load_custom_bert_from_checkpoint(ckpt_path)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path) 

    chunk = "any reasonably possible change in the foreign currency exchange rates other than disclosed above at the end of the reporting period against the respective functional currency of the entities within the group does not have a material impact on the profitloss after tax and other comprehensive incomeexpenses"
    chunk = "This section includes the exploitation of vegetal and animal natural resources, comprising the activities of growing of crops, raising and breeding of animals, harvesting of timber and other plants, animals or animal products from a farm or their natural habitats."
    chunk = """Ein Märchen ist eine alte Erzählung, die oft mit "Es war einmal..." beginnt und sich durch Merkmale wie magische Elemente, stereotype Figuren und eine oft unbestimmte Zeit und Ort auszeichnet. Bekannte Beispiele sind die Sammlung der Brüder Grimm (z.B. Hänsel und Gretel, Rotkäppchen, Aschenputtel), obwohl Märchen ursprünglich auch für Erwachsene gedacht waren und als Ratgeber für grundlegende menschliche Erfahrungen dienen können"""

    label_scores = BERT_classification_chunk(chunk, model, tokenizer)
    label_scores = {
        model.config.id2label[i]: label_scores[i].item()
        for i in range(len(label_scores))
    }
    
    #classify_report(chunks=[chunk], model=model, tokenizer=tokenizer, result_path="../../results/BERT_classification/test_1", report_path="data/datasets/stoxx_600/TXTs/Hannover Rueck SE1.txt")
