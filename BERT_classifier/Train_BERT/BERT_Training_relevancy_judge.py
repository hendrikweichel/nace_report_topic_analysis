#!/usr/bin/env python
# coding: utf-8

# In[49]:


from transformers import DataCollatorWithPadding
import torch
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
from sklearn.metrics import f1_score, precision_score, recall_score

# In[50]:


tik = time.time()

#### Run Params

train_full_model = False
model_name = "ProsusAI/finbert"
model_name = "bert-base-uncased"
num_layers = 2

#####

mypath = "/data/resources/weichel-llama3/work/projects/nace_classification/nace_report_topic_analysis/"

#data_path = "/Users/hendrikweichel/projects/NaceCodeClassification/nace_report_topic_analysis_3/data/test_read_training_data_right_classifications/paragraph_and_sentence_len_5_min_chunk_len_100_cos_thresh_0.4_nace_level_1_stoxx_with_null/"
data_path = "/Users/hendrikweichel/projects/NaceCodeClassification/nace_report_topic_analysis_3/data/test_read_training_data/paragraph_and_sentence_len_5_min_chunk_len_100_cos_thresh_0.4_nace_level_1_stoxx_with_no_class/"
data_path = "/Users/hendrikweichel/projects/NaceCodeClassification/nace_report_topic_analysis_3/data/training_data/paragraph_and_sentence_len_4_min_chunk_len_100_cos_thresh_0.4_nace_level_1_stoxx/"
data_path = "/Users/hendrikweichel/projects/NaceCodeClassification/nace_report_topic_analysis_3/data/test_read_training_data/paragraph_and_sentence_len_5_min_chunk_len_100_cos_thresh_0.4_nace_level_1_stoxx_with_no_class__cos_sim_04/"
data_path = "/Users/hendrikweichel/projects/NaceCodeClassification/nace_report_topic_analysis_3/data/training_data/dataset__reports_subset_from_full_data_1_sentence_len_6__min_chunk_len_100__cos_thresh_0.4__nace_level_1__sample_ratio_1__filter_only_right_chunks_labeling"
data_path = "../../data/training_data/dataset__reports_subset_from_full_data_1_sentence_len_6__min_chunk_len_100__cos_thresh_0.4__nace_level_1__sample_ratio_1__filter_only_right_chunks_labeling"
data_path = mypath + "data/training_data/dataset__reports_subset_from_full_data_1_sentence_len_6__min_chunk_len_100__cos_thresh_0.4__nace_level_1__sample_ratio_1__filter_only_right_chunks_labeling"
data_path = mypath + "data/training_data/dataset__reports_subset_from_full_data_1_sentence_len_6__min_chunk_len_100__cos_thresh_0.4__nace_level_1__sample_ratio_1__filter_only_right_chunks__with_null_classifiers__nace_level_1"
data_path = mypath + "data/training_data/test_dataset_sentiment"
data_path = mypath + "data/training_data/company_description"

results_path = mypath + f"results/BERT_models/Relevancy_Classifier/relevancy_judge__2__num_layers_{num_layers}__train_full_model_{train_full_model}" + os.path.basename(model_name)
#results_path = mypath + f"results/BERT_models/___results_null_classifiers__cos_thres_{new_thresh}__num_layers_{num_layers}" + os.path.basename(model_name)
    
os.makedirs(results_path, exist_ok = True)

print(results_path)

# In[51]:


# Load a custom CSV file
data_files = {"train": os.path.join(data_path, "train_data.csv"), "test": os.path.join(data_path, "test_data.csv"), "validation": os.path.join(data_path, "val_data.csv")}
# Load the CSV files using pandas
train_df = pd.read_csv(data_files["train"])
test_df = pd.read_csv(data_files["test"])
validation_df = pd.read_csv(data_files["validation"])


# In[52]:


#train_df["NACE_Code"] = train_df["NACE_Code"].apply(lambda x: False if x == "NO_CLASS" else True)
#train_df = pd.concat([train_df[train_df["NACE_Code"]].sample(len(train_df[~train_df["NACE_Code"]])),
#    train_df[~train_df["NACE_Code"]]
#           ])


# In[53]:


#test_df["NACE_Code"] = test_df["NACE_Code"].apply(lambda x: False if x == "NO_CLASS" else True)
#test_df = pd.concat([test_df[test_df["NACE_Code"]].sample(len(test_df[~test_df["NACE_Code"]])),
#    test_df[~test_df["NACE_Code"]]
#           ])


# In[54]:


#validation_df["NACE_Code"] = validation_df["NACE_Code"].apply(lambda x: False if x == "NO_CLASS" else True)
#validation_df = pd.concat([validation_df[validation_df["NACE_Code"]].sample(len(validation_df[~validation_df["NACE_Code"]])),
#    validation_df[~validation_df["NACE_Code"]]
           #])


# In[55]:


# Load a custom CSV file
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "test": Dataset.from_pandas(test_df),
    "validation": Dataset.from_pandas(validation_df)
})


# In[56]:


label_mapping = {char: val for val, char in enumerate(set(train_df["label"]))}
#dataset = dataset.map(lambda x: {"label": int(x["NACE_Code"])})


# In[57]:


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


# In[58]:


labels = dataset["train"]["label"]
class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)

label2id = {char: val for val, char in enumerate(set(dataset["test"]["label"]))}
id2label = {val: char for val, char in enumerate(set(dataset["test"]["label"]))}

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

hidden = 512

config.custom_hidden = hidden
config.custom_num_layers = num_layers

layers = []
for i in range(num_layers):
    in_dim = config.hidden_size if i == 0 else hidden
    layers.append(nn.Linear(in_dim, hidden))
    layers.append(nn.GELU())
    layers.append(nn.Dropout(0.2))

layers.append(nn.Linear(hidden, 1))

model.classifier = nn.Sequential(*layers)


# In[60]:


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
    num_train_epochs=40,              # Number of epochs
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


# In[61]:


# Load a metric (F1-score in this case)
metric = load("f1")

# Define a custom compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="weighted")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
data_collator


# In[62]:


#device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device = torch.device("cuda") if torch.backends.mps.is_available() else torch.device("cpu")


# In[66]:


import torch
import torch.nn as nn
from transformers import Trainer

class WeightedBCELossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)

        # Expecting something like [w_neg, w_pos] or a scalar for pos_weight
        if class_weights is not None:
            cw = torch.as_tensor(class_weights, dtype=torch.float32)

            if cw.numel() == 2:
                # e.g. class_weights = [w_neg, w_pos]
                # BCEWithLogitsLoss uses pos_weight (relative to negatives)
                pos_weight = cw[1] / cw[0]
            else:
                # If user already passed a single number, treat it as pos_weight directly
                pos_weight = cw.squeeze()

            self.pos_weight = pos_weight
        else:
            self.pos_weight = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # labels: [B], values {0, 1}
        labels = inputs["labels"].float()   # BCE needs float, not long

        # standard forward pass (without labels)
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits             # [B, 1] or [B]

        # Make sure shapes match: [B]
        logits = logits.view(-1)
        labels = labels.view(-1)

        # print("Logits: ", logits)
        # print("Labels: ", labels)
        
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        )
        loss = loss_fn(logits, labels)

        # print("Loss: ", loss)
        
        return (loss, outputs) if return_outputs else loss
    

trainer = WeightedBCELossTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    class_weights=class_weights,  # e.g. tensor([w_neg, w_pos])
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience=2,
        early_stopping_threshold=0.0
    )]
)


# In[ ]:


# Start trainingO
trainer.train()


# In[ ]:

################################################################################################################
#### Evaluation
################################################################################################################


# Evaluate the model
# results = trainer.evaluate()
# results_txt = str(results)
# print(results)

# Generate Threshold

val_predictions = trainer.predict(tokenized_datasets["validation"])
logits = val_predictions.predictions        
probs = torch.sigmoid(torch.tensor(logits)).numpy()
true_labels = np.array(tokenized_datasets["test"]["label"])

thresholds = np.linspace(0.01, 0.99, 99)
best_precision = -1
best_t = None

for t in thresholds:
    preds = probs > t
    preds = preds.flatten()
    
    # precision = recall_score(true_labels, preds, average="macro")
    # precision = f1_score(true_labels, preds, average="macro")
    precision = precision_score(true_labels, preds, average="macro")
    
    if precision > best_precision:
        best_precision = precision
        best_t = t

print("Best threshold:", best_t)
print("Best macro precision:", best_precision)

model.config.threshold = best_t
model.config.save_pretrained(trainer.args.output_dir)

# Generate predictions
predictions = trainer.predict(tokenized_datasets["test"])
probs = torch.sigmoid(torch.tensor(logits)).numpy()
predicted_labels = probs > best_t

# Classification report
print(classification_report(tokenized_datasets["test"]["label"], predicted_labels))
results_txt = str(classification_report(tokenized_datasets["test"]["label"], predicted_labels)) + "\n"
results_txt += "\n\n\nTime: " + str(time.time() - tik)

false_classified = ~(np.array(tokenized_datasets["test"]["label"]) == predicted_labels.flatten())

false_classified_df = pd.DataFrame(
    np.array([
        np.array(tokenized_datasets["test"]["label"])[false_classified], 
        predicted_labels[false_classified].flatten(), 
        np.array(tokenized_datasets["test"]["text"])[false_classified],        
        probs[false_classified].flatten(), 
    ]).T,
    columns=["label", "prediction", "text", "probabilities"])
        
false_classified_df["notes"] = None

false_classified_df.to_csv(os.path.join(results_path, "false_classifications.csv"))

# Confusion matrix
cm = confusion_matrix(tokenized_datasets["test"]["label"], predicted_labels, normalize="true")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_mapping.keys()))
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, xticks_rotation="vertical")
plt.show()
plt.savefig(results_path + "/conf_matrix.png")

with open(results_path + "/results_txt.txt", "w") as f:
    f.write(results_txt)

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


# In[ ]:


