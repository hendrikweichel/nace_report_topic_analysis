import sys
sys.path.append("/Users/hendrikweichel/projects/NaceCodeClassification/nace_report_topic_analysis_3")

#sys.path.append("../../..")
#from BERT_classifier.Train_BERT import BERT_Training_NO_class
from NACE_helper import NACE_code_structure

#BERT_Training_NO_class.train_BERT_model()

################################################
#### CONFIG
################################################

subtree_of_level = 2
subtree_of_class = "1"

################################################

if subtree_of_level == 1: 
    subtree_classes = NACE_code_structure.level_2[subtree_of_class]
if subtree_of_level == 2: 
    subtree_classes = NACE_code_structure.level_3[subtree_of_class]

print(subtree_classes)

# get dataset

dataset_path = 

