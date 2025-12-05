# NACE Classification 



## BERT Model Training

In BERT_classifier/Train_BERT. 

We train 2 models. A **binary relevancy filter**, with code in

- *BERT_classifier/Train_BERT/BERT_Training_relevancy_judge.py*
- *BERT_classifier/Train_BERT/BERT_Training_relevancy_judge.ipynb*
- *BERT_classifier/Train_BERT/BERT_manual_test_relevancy.ipynb* (Manual testing)

and a **NACE Classifier** with code in: 

- *BERT_classifier/Train_BERT/BERT_Training_NO_class.py *
- *BERT_classifier/Train_BERT/BERT_Training_NO_class.ipynb *
- *BERT_classifier/Train_BERT/BERT_Training_NO_class_get_threshold.py*
- *BERT_classifier/Train_BERT/BERT_manual_test.ipynb*

## Report testing

The full pipeline is implemented in *BERT_classifier/Classify_report_with_BERT/classification_report_BERT.py*. We can do tests with 

- BERT_classifier/Classify_report_with_BERT/test_reports_classification_BERT_description_pages.ipynb
- BERT_classifier/Classify_report_with_BERT/test_reports_classification_BERT.ipynb
- BERT_classifier/Classify_report_with_BERT/test_reports_classification_BERT.py

The testing environment lies in *test_base.py*, this is used for the benchmark and for the pipeline. It should also be used to test the Longformer.

Since the aggregation of logits from relevancy filter and NACE Classifier is also an important factor, we use *BERT_classifier/Classify_report_with_BERT/reevaluate_classification.ipynb* to test multiple aggregation methods.

## Benchmark Cos. Sim. 

The cosine similarity benchmark is implemented in *benchmark*. It can be run with *benchmark/test_german_annual_reports_sentence_splitting.ipynb* or *benchmark/test_german_annual_reports_sentence_splitting.py*. To create trainingdata out of the results, use

- benchmark/aggregate_training_data_reading.ipynb 
- benchmark/aggregate_training_data_2nd_approach.ipynb 

## Datasets

Datasets are stored in *data/datasets*. 

- reports_subset_from_full_data_1
    - 100 reports for each class (if available in the big dataset)
- reports_subset_from_full_data_2
- reports_subset_from_full_data_3
    - should be used for testing
    - selected 50 reports for each class (if available in the big dataset)
    - currated only reports that really contain information about the NACE class. Unfortunately, in the process of Webscraping we also received PDFs that did not contain an annual report, as ,e.g. , presentations about financials, financial reports, etc..
- stoxx_600