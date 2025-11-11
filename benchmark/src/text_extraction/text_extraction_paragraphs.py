import fitz
import tqdm
import os
import glob
import pandas as pd
import pymupdf
from text_extraction_helper import is_block_within, extract_all_sizes, extract_block_text, check_nummerical_block, check_font_sizes, heuristik_text

def get_paragraphs_without_table(pdf_path: str):

    doc = pymupdf.open(pdf_path)
    uniform_blocks = []
    temp_text = ""
    temp_font_size = 0

    for i, page in tqdm.tqdm(enumerate(doc), "Extract text without tables..."): 
        
        if i == 20: 
            break
        tabs = page.find_tables()
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if block["type"] == 0:  # Nur Textblöcke (nicht Bilder)
                if not any([is_block_within(block["bbox"], tab.bbox) for tab in tabs]):             
                    if "lines" in block.keys(): 
                        for line in block["lines"]: 
                            for span in line["spans"]:
                                if span["text"].strip(): 
                                    if span["size"] == temp_font_size:
                                        temp_text += span["text"]
                                    else:
                                        if temp_text.strip() and any(char.isalpha() for char in temp_text):
                                            temp_dict = {"text": temp_text, "page": page.number, "font_size": temp_font_size}
                                            uniform_blocks.append(temp_dict)
                                        temp_text = span["text"]
                                        temp_font_size = span["size"]

    df = pd.DataFrame(uniform_blocks)
    return df

def get_text_blocks_without_tables(pdf_path: str) -> pd.DataFrame:

    doc = pymupdf.open(pdf_path)
    uniform_blocks = []

    i = 0
    for page in doc:
        i += 1
        if i == 10: 
            break
        blocks = page.get_text("dict")["blocks"]
        tables = page.find_tables()
        for block in blocks:
            if not any([is_block_within(block["bbox"], tab.bbox) for tab in tables]) and "lines" in block.keys(): 
                block_text = ""
                block_font_sizes = []
                for line in block["lines"]: 
                    for span in line["spans"]:
                        if span["text"].strip():
                            block_text += span["text"]
                            block_font_sizes.append(span["size"])   
                            if len(set(block_font_sizes)) > 1: 
                                print(block_font_sizes)
                if block_text.strip():                                        
                    temp_dict = {"text": block_text, "page": page.number, "font_size": block_font_sizes}
                    uniform_blocks.append(temp_dict)
    
    df = pd.DataFrame(uniform_blocks)

    return df

def get_text_spans_without_tables(pdf_path: str) -> pd.DataFrame:

    doc = pymupdf.open(pdf_path)
    uniform_blocks = []

    i = 0
    for page in doc:
        i += 1
        if i == 10: 
            break
        blocks = page.get_text("dict")["blocks"]
        tables = page.find_tables()
        for block_nr, block in enumerate(blocks):
            if not any([is_block_within(block["bbox"], tab.bbox) for tab in tables]) and "lines" in block.keys(): 
                for line in block["lines"]: 
                    for span in line["spans"]:
                        span_text = ""
                        span_font_size = 0
                        if span["text"].strip():
                            span_text += span["text"]
                            span_font_size = span["size"]
                        if span_text.strip():                                        
                            temp_dict = {"text": span_text, "page": page.number, "font_size": span_font_size, "block_nr": block_nr}
                            uniform_blocks.append(temp_dict)
        
    df = pd.DataFrame(uniform_blocks)

    return df

def get_paragraphs_with_filter(path: str): 
    """ Extract all the paragraphs of the pdf and filter like this: 

    (a) Extract all blocks within the PDF with pymupdf2.
    (b) From these blocks, take all that have the mode font size in the PDF (±2%).
    (c) To also take into account those paragraphs that have a larger font size 
    but still contain relevant information, also take all text blocks that have
    more than 10 words and a ”.” character.
    (d) Remove all the blocks that come from tables (this is unnecessary data).

    Args:
        path (str): pdf path
    """

    df_spans = get_text_blocks_without_tables(path)
    df_spans["len"] = df_spans["text"].apply(len)
    df_spans["share"] = df_spans["len"] / df_spans["len"].sum()    
    font_sizes =  df_spans.groupby("font_size").sum().sort_values(by="share", ascending=False)
    max_font_size =  font_sizes.iloc[0].name

    chunks = []

    df_spans["accept"] = None

    for span_idx, span in df_spans.iterrows():
        
        text_block = span["text"]
        font_sizes_block = span["font_size"]
                                             
        if check_nummerical_block(text_block): 
            accept = False
        elif check_font_sizes(max_font_size=max_font_size, font_sizes_block=[font_sizes_block]):
            accept = True
        else: 
            accept = False

        if accept:
            chunks.append(span)

    # Aggregate Blocks 

    return chunks




if __name__ == "__main__": 

    reports = glob.glob("/Users/hendrikweichel/projects/NaceCodeClassification/nace_report_topic_analysis_3/data/PDF_stoxx600/*.pdf")

    store_at = "/Users/hendrikweichel/projects/NaceCodeClassification/nace_report_topic_analysis_3/results/chunking_analysis/reports_paragraphs_with_filters_stoxx"

    for report in tqdm.tqdm(reports): 
        chunks = get_paragraphs_with_filter(report)

        text = "\n\n".join(["\n---------------------" + str(chunk["font_size"]) + "\n\n" + chunk["text"] for chunk in chunks])

        with open(os.path.join(store_at, os.path.basename(report))+".txt", "w") as f:
            f.write(text)