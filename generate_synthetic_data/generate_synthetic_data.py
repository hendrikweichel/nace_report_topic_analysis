from sklearn.model_selection import train_test_split
import re
import json
import pandas as pd
import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm
import datetime

nace_description_path = "data/NACE_Rev2_Structure_Explanatory_Notes_EN__1_.tsv"
nace_descriptions = pd.read_csv("data/NACE_Rev2_Structure_Explanatory_Notes_EN__1_.tsv", sep="\t")

system_prompt_format = """You generate realistic paragraphs from corporate annual reports.

The text must:
- sound natural and specific
- avoid textbook or definitional language
- avoid naming industries, sectors, or classification systems
- vary structure, length, and narrative style
- include concrete operational details

The text must NOT:
- mention category names or codes
- repeat industry definitions
- follow a fixed template
- explicitly explain what the company does in generic terms

Assume the reader is familiar with the company context.
"""

few_shot_prompt_format = """Here is a definition of a industry sector:

Definition: {includes} {includes_also}

{excludes}

Here are some possible subsections:
{subsections}

Here are some examples of descriptions of these classes: 
```
{gold_standard}
```

Instruction: Write {num_samples} paragraphs (70–140 words) from a company annual report.

Context:
- Revenue depends on long-term supply contracts and fluctuating market prices
- Operations rely on land-intensive facilities and specialized equipment
- Performance is affected by weather patterns and input cost volatility
- Planning cycles are seasonal
- Activities are spread across rural regions

Constraints:
- Do not name industries, sectors, or classifications
- Do not define or explain the business in generic terms
- Avoid standard phrases used in industry descriptions
- Use a natural corporate reporting tone
"""

zero_shot_prompt_format = """Here is a definition of a industry sector:

Definition: {includes} {includes_also}

{excludes}

Here are some possible subsections:
{subsections}

Instruction: Write {num_samples} paragraphs (70–140 words) from a company annual report.

Context:
- Revenue depends on long-term supply contracts and fluctuating market prices
- Operations rely on land-intensive facilities and specialized equipment
- Performance is affected by weather patterns and input cost volatility
- Planning cycles are seasonal
- Activities are spread across rural regions

Constraints:
- Do not name industries, sectors, or classifications
- Do not define or explain the business in generic terms
- Avoid standard phrases used in industry descriptions
- Use a natural corporate reporting tone
"""
def generate_synthetic_data(
        num_samples: int, 
        gold_standard: list, 
        includes: str,
        includes_also: str, 
        excludes: str,
        subsections: list,
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
            base_url="http://10.80.20.127:11434/"
        )

    # Prompt
    if gold_standard == []: 
        #print("Zero Shot!")
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_format),
            ("human", zero_shot_prompt_format)
        ])
        gold_standard_str = ""
        
    else: 
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_format),
            ("human", few_shot_prompt_format)
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

    formatted_prompt = prompt.invoke(input)

    # Run
    response = chain.invoke(input)

    return formatted_prompt, response.content
    
def split_synthetic_data(content: str, num_samples: int): 
    content_list = content.split("\n")
    content_list = [c for c in content_list if c != ""]
    # if len(content_list) != num_samples: 
    #     print("Warning: length of creates examples != num_samples!")
    return content_list

    
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
generate_nace_class = "A"

includes = nace_descriptions[nace_descriptions["CODE"] == generate_nace_class]["Includes"].item()
assert includes is not None and includes != ""
includes_also = nace_descriptions[nace_descriptions["CODE"] == generate_nace_class]["IncludesAlso"].item()
includes_also = "" if pd.isna(includes_also) else includes_also
excludes = nace_descriptions[nace_descriptions["CODE"] == generate_nace_class]["Excludes"].item()
excludes = "" if pd.isna(excludes) else excludes

num_samples = 2
gold_standard = []
### Generate Zero-Shot Data
### Generate Few-Shot Data
# select gold standard data

# ds_2_desc  = pd.read_csv("projects/nace_classification/nace_report_topic_analysis/data/datasets/reports_subset_from_full_data_2/reports_subset_from_full_data_2_gold_standard_descriptions_for_data_generation.csv", sep=";")

# ds_2_desc  = ds_2_desc[pd.notna(ds_2_desc["Description"])]

# gold_standard = []
# # 1. take one of each lvl 3 class:
# for lvl_3 in ds_2_desc[pd.notna(ds_2_desc["Description"])].groupby("NACE_lvl_3").size().index: 
#     gold_standard.append(ds_2_desc[ds_2_desc["NACE_lvl_3"] == lvl_3].iloc[0])

# df_gold_standard = pd.concat(gold_standard, axis=1).T

#df_gold_standard.to_csv("/Users/hendrikweichel/Downloads/reports_subset_from_full_data_2_gold_standard_descriptions_for_data_generation.csv")
df_gold_standard = pd.read_csv("data/datasets/reports_subset_from_full_data_2/reports_subset_from_full_data_2_gold_standard_descriptions_for_data_generation.csv", sep=";")



# generate synthetic data for multiple groups, good night

hyperparms = [
#  {
#    "level": 2,
#    "head_nace_code": "A",
#    "generated_classes": ["1", "2", "3"]
#  },
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
#  {
#    "level": 2,
#    "head_nace_code": "B",
#    "generated_classes": ["5", "6", "7"]
#  },
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

  #{
  #  "level": 2,
  #  "head_nace_code": "C",
  #  "generated_classes": ["20", "21"]
  #},
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

  #{
  #  "level": 2,
  #  "head_nace_code": "F",
  #  "generated_classes": ["41", "42", "43"]
  #},
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

  #{
  #  "level": 2,
  #  "head_nace_code": "J",
  #  "generated_classes": ["58", "63"]
  #},
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

for h in hyperparms: 

    ##### Hyperparams
    level = h["level"]
    head_nace_code = h["head_nace_code"]
    generated_classes = h["generated_classes"]
    # generate date 

    date = datetime.datetime.now().strftime("%Y%m%d")
    store_path = "data/synthetic_data/data_" + date + f"__level_{level}__subclasses_{head_nace_code}/"
    os.makedirs(store_path, exist_ok=True)
    generated_data = {}
    num_samples = 10
    iterations_ = 100
    model = "gpt-4o-mini"

    for generate_nace_class in generated_classes:

        includes = nace_descriptions[nace_descriptions["CODE"] == generate_nace_class]["Includes"].item()
        if pd.isna(includes):
          print("No description for class:", generate_nace_class)
          continue
        includes_also = nace_descriptions[nace_descriptions["CODE"] == generate_nace_class]["IncludesAlso"].item()
        includes_also = "" if pd.isna(includes_also) else includes_also
        excludes = nace_descriptions[nace_descriptions["CODE"] == generate_nace_class]["Excludes"].item()
        excludes = "" if pd.isna(excludes) else excludes

        gold_standard = df_gold_standard[df_gold_standard["NACE_letter"] == generate_nace_class]["Description_clean"].to_list()[:3]
        gold_standard = []
        
        subsections = get_sublevels(generate_nace_class, level=2)

        examples = ""

        for i in tqdm(range(iterations_), desc=generate_nace_class):
            res = generate_synthetic_data(num_samples=num_samples, gold_standard=gold_standard, includes=includes, includes_also=includes_also, excludes=excludes, subsections=subsections, model=model)
            examples += "\n\n" + res[1]
            data = split_synthetic_data(examples, num_samples * iterations_)
            pd.DataFrame(data, columns=[generate_nace_class]).to_csv(os.path.join(store_path, f"class_{generate_nace_class}.csv"), index=False)
        
        results = {
            "data": data,
            "prompt": res[0],
            "system_prompt": res[0].messages[0].content,
            "user_prompt": res[0].messages[1].content,
            "output": examples
        }

        generated_data[generate_nace_class] = results

    
    # print prompts
    for k,v in generated_data.items(): 
        print(k,"_______"*20)
        print(v["user_prompt"])

    #### aggregate data and split
    # config

    config = {
        "prompts": {k: v["user_prompt"] for k, v in generated_data.items()}, 
        "samples": num_samples * iterations_,
        "generated_iterations": iterations_,
        "level": level,
        "head_nace_code": head_nace_code,
        "system_prompt": system_prompt_format, 
        "model": model,
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
    # make new index from 0 to len(df_full)-1
    df_full = df_full.reset_index(drop=True)

    clean_text = lambda x: re.sub(r'^\d+\.\s*', " ", x).strip()
    df_full["text"] = df_full["text"].apply(clean_text)
    df_full.to_csv(os.path.join(store_path, "synthetic_data_full.csv"), index=False)
    # make train test split 6:2:2

    train_df, temp_df = train_test_split(df_full, test_size=0.4, random_state=42, stratify=df_full["label"])
    test_df, val_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

    len(train_df), len(test_df), len(val_df)
    train_df.to_csv(os.path.join(store_path, "train_data.csv"), index=False)
    test_df.to_csv(os.path.join(store_path, "test_data.csv"), index=False)
    val_df.to_csv(os.path.join(store_path, "val_data.csv"), index=False)
    