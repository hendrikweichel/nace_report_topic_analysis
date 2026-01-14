#!/usr/bin/env python
# coding: utf-8

# In[1]:

from collections import defaultdict, deque
import pandas as pd
import os
import math
import numpy as np
import json
import random
import re
import copy
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm
import datetime

# In[2]:

if __name__ == "__main__":
    # os.chdir('/data/resources/weichel-llama3/work/projects/nace_classification/nace_report_topic_analysis')

    
    # In[5]:
    nace_description_path = "data/NACE_Rev2_Structure_Explanatory_Notes_EN__1_.tsv"
    nace_descriptions = pd.read_csv(nace_description_path, sep="\t")

def get_sublevels_df(nace_descriptions, nace_class, level) -> pd.DataFrame: 
    class_level = nace_descriptions[nace_descriptions["CODE"] == nace_class]["LEVEL"].iloc[0]
    nace_index = nace_descriptions[nace_descriptions["CODE"] == nace_class].index.item()

    i = nace_index + 1
    subtree = []

    while nace_descriptions.iloc[i]["LEVEL"] != class_level: 
        subtree.append(nace_descriptions.iloc[i])
        i += 1

    df_subtree = pd.concat(subtree, axis=1).T
    return df_subtree[df_subtree["LEVEL"] == level]

def get_zero_shot_user_prompt(path): 
    with open(path, "r") as f: 
        prompt_json = json.load(f)

    return prompt_json["user_prompt_zero_shot"]

def get_few_shot_user_prompt(path): 
    with open(path, "r") as f: 
        prompt_json = json.load(f)

    return prompt_json["user_prompt_few_shot"]

def get_system_prompt(path): 
    with open(path, "r") as f: 
        prompt_json = json.load(f)

    return prompt_json["system_prompt"]

def get_topic_prompt(path): 
    with open(path, "r") as f: 
        prompt_json = json.load(f)

    return prompt_json["user_prompt"]

# In[8]:

def generate_topic(
        num_samples: int, 
        includes: str,
        includes_also: str, 
        excludes: str,
        prompt_path: str,  
        temperature: float = 0.4, 
        model: str = "gpt-4o-mini"
): 

    # Initialize LLM
    if model == "gpt-4o-mini":
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature
        )
    elif model == "openai/gpt-oss-120b":
        llm = ChatOllama(
            model=model,
            temperature=temperature, 
            base_url="http://10.80.20.101:11434/"
        )
    elif model == "llama3.1:70b":
        llm = ChatOllama(
            model=model,
            temperature=temperature, 
            base_url="http://10.80.20.101:11434/"
        )

    user_prompt_topic = get_topic_prompt(prompt_path)

    prompt = ChatPromptTemplate.from_messages([
        ("system", ""),
        ("human", user_prompt_topic)
    ])

    # Adapt excludes
    if excludes != "": 
        excludes = "Excludes: " + excludes

    # Chain
    chain = prompt | llm

    # inputs
    input = {
        "num_samples": num_samples,
        "includes": includes,
        "includes_also": includes_also,
        "excludes": excludes,
        }

    formatted_prompt = prompt.invoke(input)

    # print("Formatted User Prompt:", formatted_prompt.messages[1].content)

    # Run
    response = chain.invoke(input)

    # print(response.content)

    return formatted_prompt, response.content


# In[9]:


def generate_synthetic_data(
        num_samples: int, 
        gold_standard: list, 
        includes: str,
        includes_also: str, 
        excludes: str,
        subsections: list,
        prompt_path: str,  
        temperature: float = 0.4, 
        model: str = "gpt-4o-mini", 
        topic: str = None
): 

    # Initialize LLM
    if model == "gpt-4o-mini":
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature
        )
    elif model == "openai/gpt-oss-120b":
        llm = ChatOllama(
            model=model,
            temperature=temperature, 
            base_url="http://10.80.20.101:11434/"
        )
    elif model == "llama3.1:70b":
        llm = ChatOllama(
            model=model,
            temperature=temperature, 
            base_url="http://10.80.20.101:11434/"
        )
    
    # Prompt
    if gold_standard == []: 
        #print("Zero Shot!")
        prompt = ChatPromptTemplate.from_messages([
            ("system", get_system_prompt(prompt_path)),
            ("human", get_zero_shot_user_prompt(prompt_path))
        ])
        gold_standard_str = ""
        
    else: 
        prompt = ChatPromptTemplate.from_messages([
            ("system", get_system_prompt(prompt_path)),
            ("human", get_few_shot_user_prompt(prompt_path))
        ])
        gold_standard_str = ""
        for i, text in enumerate(gold_standard): 
            gold_standard_str += f"Example {i+1}:\n{text}\n\n"
        gold_standard_str = gold_standard_str[:-2]

    # Subsections string
    subsections_str = "\n - " + "\n - ".join(subsections)

    # Adapt excludes
    if excludes != "": 
        excludes = "Excludes: " + excludes

    # Chain
    chain = prompt | llm

    # inputs
    input = {
        "num_samples": num_samples,
        "gold_standard": gold_standard_str,
        "includes": includes,
        "includes_also": includes_also,
        "excludes": excludes,
        "subsections": subsections_str,
        }

    # is topic used? topic in params -> business model must be in user prompt
    if topic is not None:
        if "business_model" not in prompt.input_variables: 
            print("If topic is given, prompt must conatin '{business_model}'")
            return False
        else: 
            if isinstance(topic, list): 
                topic_str = "\n".join([f"{i+1}: {text}" for i, text in enumerate(topic)])
                input["business_model"] = topic_str
            else:
                input["business_model"] = topic

    formatted_prompt = prompt.invoke(input)

    # print("Formatted Sys Prompt:", formatted_prompt.messages[0].content)
    # print("Formatted User Prompt:", formatted_prompt.messages[1].content)

    # Run
    response = chain.invoke(input)

    # print("\n\n\n", response.content)

    return formatted_prompt, response.content


# In[10]:


def split_synthetic_data(content: str, num_samples: int, only_enumarated = False): 
    content_list = content.split("\n")
    content_list = [c for c in content_list if c != ""]

    # filer out those elements that start with 1., 2., ... 
    if only_enumarated: 
        content_list = [line for line in content_list if re.match(r'^\d\s*', line)]

    # if len(content_list) != num_samples: 
    #     print("Warning: length of creates examples != num_samples!")

    return content_list

# In[11]:


def get_sublevels(nace_class, level): 
    nace_class_temp = nace_class
    nace_id = nace_descriptions[nace_descriptions["CODE"] == nace_class_temp]["ID"].iloc[0]
    nace_class_lvl_2 = []
    nace_class_lvl_3 = []
    nace_class_lvl_4 = []

    for _, row in nace_descriptions[nace_descriptions["PARENT_ID"] == nace_id].iterrows(): 
        nace_id_temp = row["ID"]
        nace_class_lvl_2.append(f'{row["NAME"]}')
        for _, row_2 in nace_descriptions[nace_descriptions["PARENT_ID"] == nace_id_temp].iterrows(): 
            nace_id_temp_temp = row_2["ID"]
            nace_class_lvl_3.append(f'{row["NAME"]}: {row_2["NAME"]}')
            for _, row_3 in nace_descriptions[nace_descriptions["PARENT_ID"] == nace_id_temp_temp].iterrows(): 
                nace_class_lvl_4.append(f'{row["NAME"]}: {row_2["NAME"]}: {row_3["NAME"]}')
        
    if level == 2: 
        return nace_class_lvl_2
    if level == 3: 
        return nace_class_lvl_3
    if level == 4: 
        return nace_class_lvl_4


if __name__ == "__main__":

    # In[14]:


    # select gold standard data

    # ds_2_desc  = pd.read_csv("projects/nace_classification/nace_report_topic_analysis/data/datasets/reports_subset_from_full_data_2/reports_subset_from_full_data_2_gold_standard_descriptions_for_data_generation.csv", sep=";")

    # ds_2_desc  = ds_2_desc[pd.notna(ds_2_desc["Description"])]

    # gold_standard = []
    # # 1. take one of each lvl 3 class:
    # for lvl_3 in ds_2_desc[pd.notna(ds_2_desc["Description"])].groupby("NACE_lvl_3").size().index: 
    #     gold_standard.append(ds_2_desc[ds_2_desc["NACE_lvl_3"] == lvl_3].iloc[0])

    # df_gold_standard = pd.concat(gold_standard, axis=1).T

    #df_gold_standard.to_csv("/Users/hendrikweichel/Downloads/reports_subset_from_full_data_2_gold_standard_descriptions_for_data_generation.csv")

    #############################################################################
    #
    # DEFINE PATHS AND HYPERPARAMS
    #
    ##############################################################################

    # In[15]:

    #df_gold_standard = pd.read_csv("projects/nace_classification/nace_report_topic_analysis/data/datasets/reports_subset_from_full_data_2/reports_subset_from_full_data_2_gold_standard_descriptions_for_data_generation.csv", sep=";")
    df_gold_standard = pd.read_csv("data/datasets/reports_subset_from_full_data_2/reports_subset_from_full_data_2_gold_standard_descriptions_for_data_generation.csv", sep=";")

    # In[16]:

    topic_prompt_path = "generate_synthetic_data/prompts/topic_prompts_1.json"   
    prompt_path = "generate_synthetic_data/prompts/prompts_5_list.json"
    # print("System Prompt: ", get_system_prompt(prompt_path))
    # print("User Prompt: ", get_few_shot_user_prompt(prompt_path))

    # In[ ]:

    few_shot = True

    hyperparms = [
    {
        "level": 1,
        "head_nace_code": None,
        "generated_classes": ["C","A"]
    #    "generated_classes": ["B", "A", "C", "J", "F"]
    },
       {
        "level": 2,
        "head_nace_code": "A",
        "generated_classes": ["1", "2", "3"]
    },
        {
      "level": 2,
      "head_nace_code": "F",
      "generated_classes": ["41", "42", "43"]
    },
        {
        "level": 2,
        "head_nace_code": "B",
        "generated_classes": ["5", "6", "7"]
      },
        {
      "level": 2,
      "head_nace_code": "C",
      "generated_classes": ["20", "21"]
    },
        {
      "level": 2,
      "head_nace_code": "J",
      "generated_classes": ["58", "63"]
    },
    {
        "level": 3,
        "head_nace_code": "1",
        "generated_classes": ["01.1", "01.2", "01.3", "01.4"]
    },
    {
        "level": 3,
        "head_nace_code": "2",
        "generated_classes": ["02.1", "02.2", "02.3"]
    },
    {
        "level": 3,
        "head_nace_code": "3",
        "generated_classes": ["03.1", "03.2"]
    },
    
    {
        "level": 3,
        "head_nace_code": "5",
        "generated_classes": ["05.1", "05.2"]
    },
    {
        "level": 3,
        "head_nace_code": "6",
        "generated_classes": ["06.1", "06.2"]
    },
    {
        "level": 3,
        "head_nace_code": "7",
        "generated_classes": ["07.1", "07.2"]
    },
    {
        "level": 3,
        "head_nace_code": "20",
        "generated_classes": ["20.1", "20.2", "20.3", "20.4", "20.5", "20.6"]
    },
    {
        "level": 3,
        "head_nace_code": "21",
        "generated_classes": ["21.1", "21.2"]
    },
    {
        "level": 3,
        "head_nace_code": "41",
        "generated_classes": ["41.1", "41.2"]
    },
    {
        "level": 3,
        "head_nace_code": "42",
        "generated_classes": ["42.1", "42.2", "42.9"]
    },
    {
        "level": 3,
        "head_nace_code": "43",
        "generated_classes": ["43.1", "43.2", "43.3", "43.9"]
    },
    {
        "level": 3,
        "head_nace_code": "58",
        "generated_classes": ["58.1", "58.2"]
    },
    {
        "level": 3,
        "head_nace_code": "63",
        "generated_classes": ["63.1", "63.9"]
    }
    ]


    n_data = 1000 # total number of samples to generate per class
    num_samples = 20 # number of samples per iteration (generated through one LLM call)
    iterations_ = n_data // num_samples
    model = "llama3.1:70b"
    model = "gpt-4o-mini"

    #############################################################################
    #
    # ITERATE OVER HYPERPARAMS
    #
    ##############################################################################

    for h in hyperparms[:1]: 
        level = h["level"]
        head_nace_code = h["head_nace_code"]
        generated_classes = h["generated_classes"]

        date = datetime.datetime.now().strftime("%Y%m%d")
        suffix = "__few_shot" if few_shot else "__zero_shot"
        suffix += "__from_lvl_4_topics"
        suffix += f"__{model}"
        store_path = "data/synthetic_data/two_step/data_" + date + f"__level_{level}__subclasses_{head_nace_code}__{os.path.basename(prompt_path).replace('.json', '')}{suffix}/" 
        os.makedirs(store_path, exist_ok=True)

        generated_data = {}

        topics = {}
    
        ################################
        # RUN DATA GENERATION FOR TOPICS
        ################################
        
        for generate_nace_class in generated_classes:

            df_subsections = get_sublevels_df(nace_descriptions, generate_nace_class, level=4)

            topics_per_subsec = math.ceil(n_data / len(df_subsections))
            topic_samples = min(topics_per_subsec, 50)
            topic_iterations = math.ceil(topics_per_subsec / topic_samples)

            print("Nbr. of Subsecs. : ", len(df_subsections), ", Topics per Subclass", topics_per_subsec, ", Samples per LLM call: ", topic_samples, ", LLM calls per Subsec.: ", topic_iterations)

            examples = []
            
            for i, subsection in tqdm(df_subsections.iterrows()): 
                includes = subsection["Includes"]
                if pd.isna(includes):
                    includes = subsection["NAME"]

                includes_also = subsection["IncludesAlso"]
                includes_also = "" if pd.isna(includes_also) else includes_also
                excludes = subsection["Excludes"]
                excludes = "" if pd.isna(excludes) else excludes

                topics_subsection = []

                for i in range(topic_iterations):
                    res = generate_topic(num_samples=topic_samples,
                                        includes=includes, 
                                        includes_also=includes_also, 
                                        excludes=excludes,  
                                        prompt_path=topic_prompt_path, 
                                        model=model, 
                                        temperature=0.8)

                    topics_subsection.append(res[1])
                
                examples.append({"CODE": subsection["CODE"], "NAME": subsection["NAME"], "TOPICS": "\n".join(topics_subsection)})
            
            examples = pd.DataFrame(examples)
            
            clean_text = lambda x: re.sub(r'^\d+\.\s*', " ", x).strip()
            data = split_synthetic_data("\n\n".join(examples["TOPICS"]), num_samples * iterations_)
            data = list(map(clean_text, data))
                
            results = {
                "topics": random.sample(data, k=2),
                "topics_df": examples,
                "system_prompt": res[0].messages[0].content,
                "user_prompt": res[0].messages[1].content,
                "output": data,
                "few_shot": few_shot,
                "model": model
            }

            topics[generate_nace_class] = results

        ############################
        # Store Topics
        ############################

        all_topics = []    
        for class_ in topics: 
            df_temp = topics[class_]["topics_df"].copy()
            df_temp["LVL_1_CODE"] = class_
            all_topics.append(df_temp)

        all_topics = pd.concat(all_topics, axis=0)
        all_topics.to_csv(os.path.join(store_path, "all_topics.csv"))

        topics_store = copy.deepcopy(topics)
        for class_ in topics_store: 
            topics_store[class_]["topics_df"] = None

        with open(os.path.join(store_path, "topics.json"), "w") as f: 
            json.dump(topics_store, f, indent=4)

        # load_topics
        # with open("data/synthetic_data/two_step/C_data_20260114__level_1__subclasses_None__prompts_5_list_hardcoded_few_shot__few_shot__from_lvl_4_topics__gpt-4o-mini/topics.json", "r") as f: 
        #     topics = json.load(f)
        
        ##################################   
        # Generate Descriptions
        ##################################   

        generated_data = {}

        for generate_nace_class in generated_classes:

            includes = nace_descriptions[nace_descriptions["CODE"] == generate_nace_class]["Includes"].item()
            if pd.isna(includes):
                print("No description for class:", generate_nace_class)
                continue
            includes_also = nace_descriptions[nace_descriptions["CODE"] == generate_nace_class]["IncludesAlso"].item()
            includes_also = "" if pd.isna(includes_also) else includes_also
            excludes = nace_descriptions[nace_descriptions["CODE"] == generate_nace_class]["Excludes"].item()
            excludes = "" if pd.isna(excludes) else excludes

            if few_shot: 
                gold_standard = df_gold_standard[df_gold_standard["NACE_letter"] == generate_nace_class]["Description_clean"].to_list()[:3] 
                gold_standard = [text.replace("\n", "") for text in gold_standard]
            else: 
                gold_standard = [] 

            subsections = get_sublevels(generate_nace_class, level=2)

            examples = []

            #if generated_data.get(generate_nace_class) is not None:
            #    if len(generated_data[generate_nace_class].get("data", [])) > 0:
            #        examples = generated_data[generate_nace_class]["data"]
            
            for i in tqdm(range(iterations_), desc=generate_nace_class):

                iteration_topics = topics[generate_nace_class]["output"][num_samples*i:num_samples*(i+1)]
                res = generate_synthetic_data(num_samples=num_samples, gold_standard=gold_standard, includes=includes, includes_also=includes_also, excludes=excludes, subsections=subsections, prompt_path=prompt_path, model=model, topic=iteration_topics)
                examples.append(res[1])
                data = split_synthetic_data("\n\n".join(examples), num_samples * iterations_, only_enumarated=False)
                pd.DataFrame(data, columns=[generate_nace_class]).to_csv(os.path.join(store_path, f"class_{generate_nace_class}.csv"), index=False)
                
            results = {
                "data": data,
                "prompt": res[0],
                "system_prompt": res[0].messages[0].content,
                "user_prompt": res[0].messages[1].content,
                "output": examples, 
                "few_shot": few_shot
            }

            generated_data[generate_nace_class] = results

        # print prompts
        for k,v in generated_data.items(): 
            print(k,len(v["output"]), "_______"*20)
            for k in v["output"]: 
                print(k)
                print("_")


        # print prompts
        for k,v in generated_data.items(): 
            print(k,"_______"*20)
            print(v["user_prompt"])


        # #### Aggregate data and split
        # config

        config = {
            "prompts": {k: v.get("user_prompt") for k, v in generated_data.items()}, 
            #"samples": num_samples * iterations_,
            #"generated_iterations": iterations_,
            "level": level,
            "head_nace_code": head_nace_code,
            "system_prompt": get_system_prompt(prompt_path), 
            "few_shot_prompting": few_shot
        }

        # store
        with open(os.path.join(store_path, "config.json"), "w") as f: 
            json.dump(config, f, indent=4)

        df_full = []
        for k, v in generated_data.items(): 
            df_temp = pd.DataFrame(v["data"], columns=["text"])
            df_temp["label"] = k
            df_full.append(df_temp)
        df_full = pd.concat(df_full, axis=0)
        df_full = df_full.reset_index(drop=True)
        df_full


        clean_text = lambda x: re.sub(r'^\d+\.\s*', " ", x).strip()

        df_full["text"] = df_full["text"].apply(clean_text)

        df_full.to_csv(os.path.join(store_path, "synthetic_data_full.csv"), index=False)

        # make train test split 6:2:2

        from sklearn.model_selection import train_test_split

        train_df, temp_df = train_test_split(df_full, test_size=0.4, random_state=42, stratify=df_full["label"])
        test_df, val_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

        train_df.to_csv(os.path.join(store_path, "train_data.csv"), index=False)
        test_df.to_csv(os.path.join(store_path, "test_data.csv"), index=False)
        val_df.to_csv(os.path.join(store_path, "val_data.csv"), index=False)
