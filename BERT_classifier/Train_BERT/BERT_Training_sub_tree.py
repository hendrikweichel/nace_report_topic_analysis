import sys
import os
import pandas as pd

wor_dir = "/data/resources/weichel-llama3/work/projects/nace_classification/nace_report_topic_analysis"
sys.path.append(wor_dir)
os.chdir(wor_dir)

#sys.path.append("../../..")
from BERT_classifier.Train_BERT import BERT_Training_NO_class
from NACE_helper import NACE_code_structure


#BERT_Training_NO_class.train_BERT_model()
################################################
#### CONFIG
################################################

subtree_of_level = 1
nace_level = subtree_of_level + 1
#subtree_of_class = "A"

data_aggregation_method = 2
dataset_version = 2
train_full_model = True
all_labels = False
model_name = "ProsusAI/finbert"
model_name = "bert-base-uncased"
num_layers = 2
new_thresh = 0.35
only_labels = False # if False also train a "no-class" class

################################################

level_1_classes = ["C", "K"]
level_1_classes = ["K"]
level_1_classes = ["A", "K"]
level_1_classes = ["C"]
level_1_classes = ["C", "K"]
level_1_classes = ["A", "C", "K"]

if subtree_of_level == 1: 
    classes = level_1_classes
elif subtree_of_level == 2: 
    classes = []
    for subtree_of_class in level_1_classes:
        subtree_classes = NACE_code_structure.level_2[subtree_of_class]    
        classes.extend(subtree_classes)

print(f"Iterate over {classes}")

set1 = [['F', ['41', '42']], 
        ['J', ['58', '61']], 
        ['K', ['64', '66']], 
        ['N', ['77', '81']]]

set2 = [['77', ['77.1', '77.3']],
        ['81', ['81.2', '81.3']], 
        ['41', ['41.1', '41.2']], 
        ['42', ['42.1', '42.2', "42.9"]], 
        ['58', ['58.1', '58.2']], 
        ['61', ['61.1', '61.2']],
        ['64', ['64.1', '64.2', '64.3', '64.9']],
        ['66', ['66.1', '66.2', '66.3']]]

#for subtree_of_class in classes:
for subtree_of_class, take_classes in set1:
    
    if subtree_of_level == 1: 
        subtree_classes = NACE_code_structure.level_2[subtree_of_class]
    if subtree_of_level == 2: 
        subtree_classes = NACE_code_structure.level_3[subtree_of_class]

    if not only_labels:
        subtree_classes.append("NO_CLASS")
        take_classes.append("NO_CLASS")
    
    # Print what we now train
    print(f"Training Model for level {nace_level} and level 1 classes {level_1_classes}. Train subtree of {subtree_of_class} -> {subtree_classes}")
    
    # get dataset
    
    mypath = "/data/resources/weichel-llama3/work/projects/nace_classification/nace_report_topic_analysis/"
                    #data_path = "/Users/hendrikweichel/projects/NaceCodeClassification/nace_report_topic_analysis_3/data/test_read_training_data_right_classifications/paragraph_and_sentence_len_5_min_chunk_len_100_cos_thresh_0.4_nace_level_1_stoxx_with_null/"
    
    data_path = mypath + f"data/training_data/approach_{data_aggregation_method}/dataset__reports_subset_from_full_data_{dataset_version}_sentence_len_6__min_chunk_len_100__cos_thresh_0.4__nace_level_{nace_level}__2nd_approach__nace_level_{nace_level}__cos_thres_{new_thresh}"
    dataset_description = f"data_approach_{data_aggregation_method}"
    
    results_path = mypath + f"results/BERT_models/NACE_classification/NACE_level_{nace_level}/NACE_class_{subtree_of_class}/"
                    
    experiment_nbr = BERT_Training_NO_class.get_experiment_nbr(results_path)
    results_path = results_path + f"{experiment_nbr}_results__{dataset_description}__num_layers_{num_layers}__cos_thres_{new_thresh}" + os.path.basename(model_name)
    
    if train_full_model:
        results_path += "__train_full_model" 
    else: 
        results_path += "__train_classifier_only" 
    
    if all_labels: 
        results_path += "__all_labels" 
    else: 
        results_path += "__some_labels_no_G" 
    if only_labels: 
        results_path += "__only_labels" 
    else: 
        results_path += "__with_no_class" 
    
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
        #train_df = train_df[train_df["NACE_Code"]!="R"]
        #train_df = train_df[train_df["NACE_Code"]!="S"]
        #train_df = train_df[train_df["NACE_Code"]!="T"]
        #train_df = train_df[train_df["NACE_Code"]!="N"]
        #train_df = train_df[train_df["NACE_Code"]!="G"]
        train_df = train_df.reset_index(drop=True)
        
        #test_df = test_df[test_df["NACE_Code"]!="C"]
        #test_df = test_df[test_df["NACE_Code"]!="N"]
        #test_df = test_df[test_df["NACE_Code"]!="P"]
        #test_df = test_df[test_df["NACE_Code"]!="M"]
        #test_df = test_df[test_df["NACE_Code"]!="G"]
        #test_df = test_df[test_df["NACE_Code"]!="N"]
        #test_df = test_df[test_df["NACE_Code"]!="S"]
        #test_df = test_df[test_df["NACE_Code"]!="R"]
        #test_df = test_df[test_df["NACE_Code"]!="T"]
        test_df = test_df.reset_index(drop=True)
        
        #validation_df = validation_df[validation_df["NACE_Code"]!="C"]
        #validation_df = validation_df[validation_df["NACE_Code"]!="P"]
        #validation_df = validation_df[validation_df["NACE_Code"]!="M"]
        #validation_df = validation_df[validation_df["NACE_Code"]!="N"]
        #validation_df = validation_df[validation_df["NACE_Code"]!="G"]
        #validation_df = validation_df[validation_df["NACE_Code"]!="R"]
        #validation_df = validation_df[validation_df["NACE_Code"]!="N"]
        #validation_df = validation_df[validation_df["NACE_Code"]!="S"]
        #validation_df = validation_df[validation_df["NACE_Code"]!="T"]
        validation_df = validation_df.reset_index(drop=True)
    
    if only_labels:
        train_df = train_df[train_df["NACE_Code"]!="NO_CLASS"]
        test_df = test_df[test_df["NACE_Code"]!="NO_CLASS"]
        validation_df = validation_df[validation_df["NACE_Code"]!="NO_CLASS"]
    # Filter subtree
    
    train_df  = train_df[train_df["NACE_Code"].apply(lambda x: x in subtree_classes)]
    test_df  = test_df[test_df["NACE_Code"].apply(lambda x: x in subtree_classes)]
    validation_df  = validation_df[validation_df["NACE_Code"].apply(lambda x: x in subtree_classes)]

    count_classes_train = train_df.groupby("NACE_Code").count()["Score"].to_dict()
    #take_classes = [k for k, v in count_classes_train.items() if v > 1]

    print(f"Train classes: {take_classes}, drop: {set(train_df['NACE_Code'])-set(take_classes)}")
    print(train_df.groupby("NACE_Code").count()["Score"].to_dict())
    print(train_df.head())
    
    train_df = train_df[train_df["NACE_Code"].apply(lambda x: x in take_classes)]
    test_df = test_df[test_df["NACE_Code"].apply(lambda x: x in take_classes)]
    validation_df = validation_df[validation_df["NACE_Code"].apply(lambda x: x in take_classes)]

    if not only_labels:
        amount_no_class = test_df[test_df["NACE_Code"] != "NO_CLASS"]["NACE_Code"].value_counts().max()
        test_df_right = test_df[test_df["NACE_Code"] != "NO_CLASS"]
        test_df_NO_CLASS = test_df.loc[test_df["NACE_Code"] == "NO_CLASS"].sample(n=amount_no_class)
        test_df = pd.concat([test_df_right, test_df_NO_CLASS], axis=0)
        
        amount_no_class = train_df[train_df["NACE_Code"] != "NO_CLASS"]["NACE_Code"].value_counts().max()
        train_df_right = train_df[train_df["NACE_Code"] != "NO_CLASS"]
        train_df_NO_CLASS = train_df.loc[train_df["NACE_Code"] == "NO_CLASS"].sample(n=amount_no_class)
        train_df = pd.concat([train_df_right, train_df_NO_CLASS], axis=0)
        
        amount_no_class = validation_df[validation_df["NACE_Code"] != "NO_CLASS"]["NACE_Code"].value_counts().max()
        validation_df_right = validation_df[validation_df["NACE_Code"] != "NO_CLASS"]
        validation_df_NO_CLASS = validation_df.loc[validation_df["NACE_Code"] == "NO_CLASS"].sample(n=amount_no_class)
        validation_df = pd.concat([validation_df_right, validation_df_NO_CLASS], axis=0)
    
    if len(train_df) < 10:
        continue    
    if "NACE_Code" in train_df.columns: 
        if len(train_df.groupby("NACE_Code").count()["Score"].to_dict()) < 2: 
            continue    

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
        "train_distribution": train_df.groupby("NACE_Code").count()["Score"].to_dict(),
        "test_distribution": test_df.groupby("NACE_Code").count()["Score"].to_dict(),
        "validation_distribution": validation_df.groupby("NACE_Code").count()["Score"].to_dict(),
    }

    BERT_Training_NO_class.train_BERT_model(
        train_df = train_df,
        test_df = test_df,
        validation_df = validation_df,
        results_path = results_path,
        train_full_model = train_full_model,
        model_name = model_name,
        num_layers = num_layers,
        training_config = training_config, 
    )
