#!/usr/bin/env python
# coding: utf-8

# In[]:

from transformers import DataCollatorWithPadding
import torch
import json
import torch.nn as nn
from transformers import Trainer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch
from datasets import load_dataset
import os
from transformers import AutoTokenizer, EarlyStoppingCallback
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoConfig
from evaluate import load
import matplotlib.pyplot as plt
from torch import nn
import pandas as pd
from datasets import DatasetDict, Dataset
import time 
from safetensors.torch import load_file  # comes with HF if safetensors installed
from scipy.special import softmax
# In[]:

tik = time.time()

def get_experiment_nbr(base_dir="results"):

    # List existing experiment folders
    existing = [
        d[:3] for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d[:3].isdigit()
    ]

    # Determine next experiment number
    if existing:
        next_num = max(int(d) for d in existing) + 1
    else:
        next_num = 1

    # Format as 3 digits with leading zeros
    folder_name = f"{next_num:03d}"

    return folder_name

    
for dataset_nbr in [2]:
    for num_layers in [1,2,3]:
    
        #thres = 0.4
        description_class = 1
        #for thres in [0.35,0.4,0.45,0.5,0.55]:
        #for description_class in [1,2,3,4]:
        for thres in [0.4,0.45,0.5]:
            #### Run Params
    
            train_full_model = True
            all_labels = False
            model_name = "ProsusAI/finbert"
            model_name = "bert-base-uncased"
            #num_layers = 2
            new_thresh = thres
            only_labels = False # if False also train a "no-class" class
            
            #####
            
            mypath = "/data/resources/weichel-llama3/work/projects/nace_classification/nace_report_topic_analysis/"
            #data_path = "/Users/hendrikweichel/projects/NaceCodeClassification/nace_report_topic_analysis_3/data/test_read_training_data_right_classifications/paragraph_and_sentence_len_5_min_chunk_len_100_cos_thresh_0.4_nace_level_1_stoxx_with_null/"

            data_path = mypath + f"data/training_data/approach_2/dataset__reports_subset_from_full_data_2_sentence_len_6__min_chunk_len_100__cos_thresh_0.4__nace_level_1__2nd_approach__nace_level_1__cos_thres_0.4"
            dataset_description = f"data_approach_2__desc_lvl_level_{description_class}"
            
            data_path = mypath + f"data/training_data/approach_3/dataset__reports_subset_from_full_data_2_sentence_len_6__min_chunk_len_100__cos_thresh_0.4__nace_level_1__sample_ratio_1__filter_only_right_chunks__with_null_classifiers__nace_level_1__cos_thres_0.4"
            dataset_description = f"data_approach_3__desc_lvl_level_{description_class}"
            
            results_path = mypath + f"results/BERT_models/NACE_classification/"
            
            experiment_nbr = get_experiment_nbr(results_path)
            results_path = results_path + f"{experiment_nbr}_results__{dataset_description}__num_layers_{num_layers}__cos_thres_{new_thresh}" + os.path.basename(model_name)
            
            if train_full_model:
                results_path += "__train_full_model" 
            else: 
                results_path += "__train_classifier_only" 
            
            if all_labels: 
                results_path += "__all_labels" 
            else: 
                results_path += "__some_labels" 
    
            if only_labels: 
                results_path += "__only_labels" 
            else: 
                results_path += "__with_no_class" 
            
            os.makedirs(results_path, exist_ok = True)
            
            # In[]:
            
            # Load a custom CSV file
            data_files = {"train": os.path.join(data_path, "train_data.csv"), "test": os.path.join(data_path, "test_data.csv"), "validation": os.path.join(data_path, "val_data.csv")}
            # Load the CSV files using pandas
            train_df = pd.read_csv(data_files["train"])
            test_df = pd.read_csv(data_files["test"])
            validation_df = pd.read_csv(data_files["validation"])
            
            train_df = train_df[(train_df["Score"] > new_thresh) | (train_df["NACE_Code"] == "NO_CLASS")]
            validation_df = validation_df[(validation_df["Score"] > new_thresh) | (test_df["NACE_Code"] == "NO_CLASS")]
            test_df = test_df[(test_df["Score"] > new_thresh) | (test_df["NACE_Code"] == "NO_CLASS")]
            
            
            if not all_labels: 
                #train_df = train_df[train_df["NACE_Code"]!="C"]
                #train_df = train_df[train_df["NACE_Code"]!="P"]
                #train_df = train_df[train_df["NACE_Code"]!="M"]
                #train_df = train_df[train_df["NACE_Code"]!="M"]
                train_df = train_df[train_df["NACE_Code"]!="R"]
                train_df = train_df[train_df["NACE_Code"]!="S"]
                train_df = train_df[train_df["NACE_Code"]!="T"]
                train_df = train_df[train_df["NACE_Code"]!="N"]
                train_df = train_df.reset_index(drop=True)
                
                #test_df = test_df[test_df["NACE_Code"]!="C"]
                #test_df = test_df[test_df["NACE_Code"]!="N"]
                #test_df = test_df[test_df["NACE_Code"]!="P"]
                #test_df = test_df[test_df["NACE_Code"]!="M"]
                test_df = test_df[test_df["NACE_Code"]!="N"]
                test_df = test_df[test_df["NACE_Code"]!="S"]
                test_df = test_df[test_df["NACE_Code"]!="R"]
                test_df = test_df[test_df["NACE_Code"]!="T"]
                test_df = test_df.reset_index(drop=True)
                
                #validation_df = validation_df[validation_df["NACE_Code"]!="C"]
                #validation_df = validation_df[validation_df["NACE_Code"]!="P"]
                #validation_df = validation_df[validation_df["NACE_Code"]!="M"]
                #validation_df = validation_df[validation_df["NACE_Code"]!="N"]
                validation_df = validation_df[validation_df["NACE_Code"]!="R"]
                validation_df = validation_df[validation_df["NACE_Code"]!="N"]
                validation_df = validation_df[validation_df["NACE_Code"]!="S"]
                validation_df = validation_df[validation_df["NACE_Code"]!="T"]
                validation_df = validation_df.reset_index(drop=True)
    
            if only_labels:
                train_df = train_df[train_df["NACE_Code"]!="NO_CLASS"]
                test_df = test_df[test_df["NACE_Code"]!="NO_CLASS"]
                validation_df = validation_df[validation_df["NACE_Code"]!="NO_CLASS"]
    
            
            # Load a custom CSV file
            dataset = DatasetDict({
                "train": Dataset.from_pandas(train_df),
                "test": Dataset.from_pandas(test_df),
                "validation": Dataset.from_pandas(validation_df)
            })
            
            #label_mapping = {char: val for val, char in enumerate(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U',"NO_CLASS"])}
            label_mapping = {char: val for val, char in enumerate(set(train_df["NACE_Code"]))}
            dataset = dataset.map(lambda x: {"label": label_mapping[x["NACE_Code"]]})
            
            # Initialize the BERT tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Tokenize a sample text
            sample_text = dataset["train"][0]["text"]
            tokens = tokenizer(sample_text, padding="max_length", truncation=True, max_length=512)
            
            
            def tokenize_function(examples):
                return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
            
            # Apply the tokenizer to the dataset
            tokenized_datasets = dataset.map(tokenize_function, batched=True)
            
            # Inspect tokenized samples
            print(tokenized_datasets["train"][0])
            
            labels = dataset["train"]["label"]
            class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
            
            label2id = {char: val for val, char in enumerate(set(dataset["test"]["NACE_Code"]))}
            id2label = {val: char for val, char in enumerate(set(dataset["test"]["NACE_Code"]))}
            
            if model_name == "ProsusAI/finbert": 
                config = AutoConfig.from_pretrained(
                    model_name, 
                    num_labels=len(set(labels)), 
                    problem_type="single_label_classification", 
                    id2label=id2label,
                    label2id=label2id
                    )
                model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
            elif model_name == "bert-base-uncased": 
                config = AutoConfig.from_pretrained(
                    model_name, 
                    num_labels=len(set(labels)), 
                    id2label=id2label,
                    label2id=label2id
                    )
                model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
            print(model.config)
            
            hidden = 512
            
            config.custom_hidden = hidden
            config.custom_num_layers = num_layers
            
            layers = []
            for i in range(num_layers):
                in_dim = config.hidden_size if i == 0 else hidden
                layers.append(nn.Linear(in_dim, hidden))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(0.2))
            
            layers.append(nn.Linear(hidden, config.num_labels))
            
            model.classifier = nn.Sequential(*layers)
            
            # In[107]:
            
            for param in model.bert.parameters():
                param.requires_grad = train_full_model
                #param.requires_grad = True # Train the wmbeddings as well!
            
            # Keep only the classification head trainable
            for param in model.classifier.parameters():
                param.requires_grad = True
            
            print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            
            # Define training arguments
            training_args = TrainingArguments(
                output_dir=results_path,          # Directory for saving model checkpoints
                evaluation_strategy="epoch",     # Evaluate at the end of each epoch
                learning_rate=5e-5,              # Start with a small learning rate
                per_device_train_batch_size=16,  # Batch size per GPU
                per_device_eval_batch_size=16,
                num_train_epochs=1,              # Number of epochs
                weight_decay=0.01,               # Regularization
                save_total_limit=1,              # Limit checkpoints to save space
                load_best_model_at_end=True,     # Automatically load the best checkpoint
                logging_dir="./logs",            # Directory for logs
                logging_strategy="epoch",            # Directory for logs
                logging_steps=100,               # Log every 100 steps
                #fp16=True,                      # Enable mixed precision for faster training
                save_strategy="epoch"
            )
            
            print(training_args)
            
            
            # In[109]:
            
            
            # Load a metric (F1-score in this case)
            metric = load("f1")
            
            # Define a custom compute_metrics function
            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                predictions = logits.argmax(axis=-1)
                return metric.compute(predictions=predictions, references=labels, average="weighted")
            
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            data_collator
            
            # In[111]:
            
            #device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
            device = torch.device("cuda") if torch.backends.mps.is_available() else torch.device("cpu")
            
            # In[112]:
            
            class WeightedCELossTrainer(Trainer):
                def __init__(self, *args, class_weights=None, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.class_weights = (
                        torch.as_tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
                    )
            
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    labels = inputs["labels"]                 # shape [B], dtype long
                    outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
                    logits  = outputs.logits                  # [B, C]
                    loss_fn = nn.CrossEntropyLoss(
                        weight=self.class_weights.to(logits.device) if self.class_weights is not None else None
                    )
                    loss = loss_fn(logits, labels)
                    return (loss, outputs) if return_outputs else loss
            
            # load model
            #ckpt_path = "results/BERT_models/results__new_approach_data__num_layers_2__cos_thres_0.35bert-base-uncased__train_full_model__some_labels/checkpoint-15681"
            #state_dict = load_file(os.path.join(ckpt_path, "model.safetensors"))
            #model.load_state_dict(state_dict, strict=True)  # will fail loudly if mismatch
            
            
            trainer = WeightedCELossTrainer(
                model=model,                        # Pre-trained BERT model
                args=training_args,                 # Training arguments
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                tokenizer=tokenizer,
                data_collator=data_collator,        # Efficient batching
                class_weights=class_weights,   # <— here’s A
                compute_metrics=compute_metrics,     # Custom metric
                callbacks=[EarlyStoppingCallback(
                    early_stopping_patience=2,      # stop after 2 evals with no improvement
                    early_stopping_threshold=0.0    # optional min improvement; e.g. 1e-4
                )]
            )
    
            ### Store config
    
            training_config = {
                "data_path" : data_path,
                "train_full_model" : train_full_model,
                "all_labels": all_labels,
                "model_name" : model_name,
                "num_layers" : num_layers,
                "new_thresh" : new_thresh,
                "only_labels": only_labels, 
                "len_test": len(test_df),
                "len_train": len(train_df),
                "len_val": len(validation_df), 
                "test_distribution": test_df.groupby("NACE_Code").count()["Score"].to_dict() 
            }
            with open(os.path.join(results_path, "training_config.json"), "w") as f:
                json.dump(training_config, f)
            
    
            # In[]:
            
            # Start trainingO
            trainer.train()
            
            # In[]:
            
            # Evaluate the model
            results = trainer.evaluate()
            
            results_txt = str(results)
            
            print(results)
    
            
            # In[]:
            
            # Generate predictions
            predictions = trainer.predict(tokenized_datasets["test"])
            predicted_labels = predictions.predictions.argmax(axis=-1)
            
            # Classification report
            print(classification_report(tokenized_datasets["test"]["label"], predicted_labels))
            
            results_txt += str(classification_report(tokenized_datasets["test"]["label"], predicted_labels))
            results_txt += "\n\n\nTime: " + str(time.time() - tik)
            
            # Confusion matrix
            cm = confusion_matrix(tokenized_datasets["test"]["label"], predicted_labels, normalize="true")
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_mapping.keys()))
            fig, ax = plt.subplots(figsize=(10, 10))
            disp.plot(ax=ax, xticks_rotation="vertical")
            plt.show()
            plt.savefig(results_path + "/conf_matrix.png")
            
            with open(results_path + "/results_txt.txt", "w") as f:
                f.write(results_txt)
            
            with open(results_path + "/results.json", "w") as f:
                json.dump(classification_report(tokenized_datasets["test"]["label"], predicted_labels, output_dict=True), f)
    
            # create false classified dataset
            false_classified = ~(np.array(tokenized_datasets["test"]["label"]) == predicted_labels)
    
            false_classified_df = pd.DataFrame(
                np.array([
                    np.array(tokenized_datasets["test"]["label"])[false_classified], 
                    predicted_labels[false_classified], 
                    np.array(tokenized_datasets["test"]["text"])[false_classified],        
                ]).T,
                columns=["label", "prediction", "text"],)
            
            false_classified_df["label"] = false_classified_df["label"].apply(lambda x: id2label[int(x)])
            false_classified_df["prediction"] = false_classified_df["prediction"].apply(lambda x: id2label[int(x)])
            
            false_classified_df["probabilities"] = false_classified_df["prediction"].apply(
                lambda pred: dict(sorted(
                    {id2label[i]: p for i, p in enumerate(softmax(predictions.predictions[false_classified][np.where(false_classified_df["prediction"] == pred)[0][0]]))}.items(),
                    key=lambda item: item[1], reverse=True
                ))
            )
            false_classified_df["notes"] = None
            false_classified_df.to_csv(os.path.join(results_path, "false_classifications.csv"))
    
            logs = pd.DataFrame(trainer.state.log_history)
            logs.head()
            
            plt.figure(figsize=(8,5))
            
            plt.figure(figsize=(8,5))
            logs_epoch = logs.dropna(subset="loss")
            eval_loss_epoch = logs.dropna(subset="eval_loss")
            plt.plot(logs_epoch["epoch"], logs_epoch["loss"], label="Training Loss")
            if "eval_loss" in logs:
                plt.plot(eval_loss_epoch["epoch"], eval_loss_epoch["eval_loss"], label="Validation Loss")
            
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Loss per Epoch")
            plt.legend()
            plt.savefig(results_path + "/losses.png")
    
            # get threshold for classification
            
            logits = predictions.predictions          # shape: (N, num_labels)
            probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
            true_labels = np.array(tokenized_datasets["test"]["label"])
    
            from sklearn.metrics import f1_score
    
            thresholds = np.linspace(0.01, 0.99, 99)
            best_f1 = -1
            best_t = None
            
            for t in thresholds:
                preds = []
                for p in probs:
                    max_prob = p.max()
                    pred_class = p.argmax()
            
                    if max_prob < t:
                        preds.append(-1)  # -1 = NO_CLASS oder Reject
                    else:
                        preds.append(pred_class)
            
                f1 = f1_score(true_labels, preds, average="macro")
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
    
            print("Best threshold:", best_t)
            print("Best macro F1:", best_f1)