from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
from transformers import Trainer
import torch
from datasets import load_dataset
import os
from transformers import AutoTokenizer
from sklearn.utils.class_weight import compute_class_weight
from transformers import DataCollatorWithPadding
import numpy as np
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoConfig
from evaluate import load
import matplotlib.pyplot as plt
from torch import nn


#data_path = "/Users/hendrikweichel/projects/NaceCodeClassification/nace_report_topic_analysis_3/data/test_read_training_data_right_classifications/paragraph_and_sentence_len_5_min_chunk_len_100_cos_thresh_0.4_nace_level_1_stoxx_with_null/"
# data_path = "/Users/hendrikweichel/projects/NaceCodeClassification/nace_report_topic_analysis_3/data/test_read_training_data/paragraph_and_sentence_len_5_min_chunk_len_100_cos_thresh_0.4_nace_level_1_stoxx_with_no_class/"
# data_path = "/Users/hendrikweichel/projects/NaceCodeClassification/nace_report_topic_analysis_3/data/training_data/paragraph_and_sentence_len_4_min_chunk_len_100_cos_thresh_0.4_nace_level_1_stoxx/"
# data_path = "/Users/hendrikweichel/projects/NaceCodeClassification/nace_report_topic_analysis_3/data/test_read_training_data/paragraph_and_sentence_len_5_min_chunk_len_100_cos_thresh_0.4_nace_level_1_stoxx_with_no_class__cos_sim_04/"
# data_path = "/Users/hendrikweichel/projects/NaceCodeClassification/nace_report_topic_analysis_3/data/test_read_training_data/paragraph_and_sentence_len_5_min_chunk_len_100_cos_thresh_0.4_nace_level_1_stoxx_with_no_class__cos_sim_04/"
data_path = "./"


# Load a custom CSV file
data_files = {"train": os.path.join(data_path, "train_data.csv"), "test": os.path.join(data_path, "test_data.csv"), "validation": os.path.join(data_path, "val_data.csv")}
dataset = load_dataset("csv", data_files=data_files)
label_mapping = {char: val for val, char in enumerate(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U',"NO_CLASS"])}
dataset = dataset.map(lambda x: {"label": label_mapping[x["NACE_Code"]]})

# Initialize the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize a sample text
sample_text = dataset["train"][0]["text"]
tokens = tokenizer(sample_text, padding="max_length", truncation=True, max_length=128)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Apply the tokenizer to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Inspect tokenized samples
labels = dataset["train"]["label"]
class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)

# Initialize a BERT model for binary classification
model_name = "bert-base-uncased"
config = AutoConfig.from_pretrained(model_name, num_labels=len(set(labels)))
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

hidden = 512
model.classifier = nn.Sequential(
    nn.Linear(config.hidden_size, hidden),
    nn.GELU(),
    nn.Dropout(0.2),
    nn.Linear(hidden, config.num_labels),
)
for param in model.bert.parameters():
    param.requires_grad = False
    #param.requires_grad = True # Train the wmbeddings as well!

# Keep only the classification head trainable
for param in model.classifier.parameters():
    param.requires_grad = True

print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",          # Directory for saving model checkpoints
    evaluation_strategy="epoch",     # Evaluate at the end of each epoch
    learning_rate=5e-5,              # Start with a small learning rate
    per_device_train_batch_size=16,  # Batch size per GPU
    per_device_eval_batch_size=16,
    use_mps_device=True,             # <— key switch
    num_train_epochs=20,              # Number of epochs
    weight_decay=0.01,               # Regularization
    save_total_limit=2,              # Limit checkpoints to save space
    load_best_model_at_end=True,     # Automatically load the best checkpoint
    logging_dir="./logs",            # Directory for logs
    logging_steps=100,               # Log every 100 steps
    #fp16=True,                      # Enable mixed precision for faster training
    save_strategy="epoch"
)

print(training_args)

# Load a metric (F1-score in this case)
metric = load("f1")

# Define a custom compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="weighted")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

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

trainer = WeightedCELossTrainer(
    model=model,                        # Pre-trained BERT model
    args=training_args,                 # Training arguments
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,        # Efficient batching
    class_weights=class_weights,   # <— here’s A
    compute_metrics=compute_metrics     # Custom metric
)

# Start trainingO
trainer.train()

# Evaluate the model
results = trainer.evaluate()

# Generate predictions
predictions = trainer.predict(tokenized_datasets["test"])
predicted_labels = predictions.predictions.argmax(axis=-1)

# Classification report
cm = confusion_matrix(tokenized_datasets["test"]["label"], predicted_labels, normalize="true")
alphabet = "ABCDEFGHIJKLMNOPQRSTUVW"
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"Label_{alphabet[i]}" for i in range(0, 21)])
fig, ax = plt.subplots(figsize=(30, 10))  # <-- set size here
disp.plot(ax=ax, xticks_rotation="vertical")
ckpt = "results/checkpoint-5668"
final_dir = "my_model_final"  
plt.savefig(final_dir + "confusion_matrix")

model.save_pretrained(final_dir)         # writes config + model.safetensors (or pytorch_model.bin)
tokenizer.save_pretrained(final_dir)     # writes tokenizer files
trainer.log_metrics("eval", final_dir)   # prints nicely & sends to logger
trainer.save_metrics("eval", final_dir)

# model = AutoModelForSequenceClassification.from_pretrained(ckpt)
# tokenizer = AutoTokenizer.from_pretrained(ckpt)

# sentenec = "all manufacturing of secondary aluminium is defined by the taxonomy as making a substantial contribution to climate change mitigation. to be a taxonomyaligned activity the manufacture of secondary aluminium must also comply with the dnsh criteria for manufacture of aluminium and hydro must comply with the criteria for processes and outcomes related to human rights bribery and corruption taxation and fair competition minimum safeguards."
# sentenec = "all manufacturing of secondary aluminium is defined by the taxonomy as making a substantial contribution to climate change mitigation. to be a taxonomyaligned activity the manufacture of secondary aluminium must also comply with the dnsh criteria for manufacture of aluminium and hydro must comply with the criteria for processes and outcomes related to human rights bribery and corruption taxation and fair competition minimum safeguards."
# sentenec = "i. regasification basic and secondary transmission as well as storage of natural gas via the corresponding gas infrastructure or facilities of its own or of third parties and also the performance of auxiliary activities or others related to the aforementioned activities."
# inputs = tokenizer(sentenec, return_tensors="pt")
# with torch.no_grad():
#     outputs = model(**inputs)
#     logits = outputs.logits
#     predictions = torch.argmax(logits, dim=-1)
# # If you set id2label in config earlier, you can get human-readable labels
# label = model.config.id2label[predictions.item()]
# print("Predicted label:", label)

