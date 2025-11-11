from typing import List
import re
import tqdm
import pandas as pd
import pymupdf
from langchain.document_loaders import PyPDFLoader 
from langchain.text_splitter import CharacterTextSplitter

def is_block_within(block1, block2):
    """
    Returns True if block1 is fully inside block2's bounding box.
    Each block must have a 'bbox' key: [x0, y0, x1, y1]
    """
    x0_1, y0_1, x1_1, y1_1 = block1
    x0_2, y0_2, x1_2, y1_2 = block2
    
    return (
        x0_1 >= x0_2 and
        y0_1 >= y0_2 and
        x0_1 <= x1_2 and
        y0_1 <= y1_2
    )

def get_text_from_pdf(pdf_path: str) -> str: 
    loader = PyPDFLoader(pdf_path)
    seiten_docs = loader.load()
    text = " ".join([page.page_content for page in seiten_docs])

    return text

def preprocess_text(text: str) -> str: 
    text = text.replace(" -", "")
    text = text.replace("\n", "")
    text = text.replace("\t", "")
    
    # only keep conventional letters
    text = re.sub(r"[^a-zA-ZäöüÄÖÜß.\s]", "", text)
    # delete multiple consecutive spaces
    text = re.sub(r' +', ' ', text)

    text = text.strip()
    text = text.lower()

    return text
    
def get_sliding_window(text: str, window_len: int = 3) -> List[str]: 

    sentences = text.split(".")
    sentences = [". ".join(sentences[i:i+window_len+1]) for i in range(len(sentences)-window_len)]
    sentences = [sentence.lower().strip() for sentence in sentences]

    return sentences

def get_paragraphs(pdf_path: str) -> List[str]:

    doc = pymupdf.open(pdf_path)
    uniform_blocks = []
    temp_text = ""
    temp_font_size = 0

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
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

    return df["text"].to_list()


def get_paragraphs_without_table(pdf_path: str) -> List[str]:

    doc = pymupdf.open(pdf_path)
    uniform_blocks = []
    temp_text = ""
    temp_font_size = 0

    for i, page in tqdm.tqdm(enumerate(doc), "Extract text without tables..."): 
            
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
    return df["text"].to_list()


def get_paragraphs_without_table_only_mode_font_size(pdf_path: str) -> pd.DataFrame:

    doc = pymupdf.open(pdf_path)
    uniform_blocks = []
    temp_text = ""
    temp_font_size = 0

    for i, page in tqdm.tqdm(enumerate(doc)): 
            
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

    df["len"] = df["text"].apply(len)
    df["share"] = df["len"] / df["len"].sum()    
    mode_font_size = df.groupby("font_size").sum().sort_values(by="share", ascending=False).iloc[0].name

    df = df[df["font_size"] == mode_font_size]

    return df

def get_chunked_paragraphs(pdf_path: str, chunk_length: int) -> List[str]: 
    
    paragraphs = get_paragraphs_without_table(pdf_path=pdf_path)

    chunks = []

    for paragraph in paragraphs: 
        sentences = paragraph.split(".")
        
        sentences = [s for s in sentences if any(char.isalpha() for char in s)]
        sentences = [s for s in sentences if s != " "]
        sentences = [s for s in sentences if s != ""]

        sentences = [re.sub(r'\b\d+\.\d+\b', '', s) for s in sentences]

        paragraph_chunks = []
        i = 0
        
        while i < len(sentences):
            paragraph_chunks.append((". ".join(sentences[i:i+chunk_length]) + ".").strip())
            i += chunk_length 
        
        chunks += paragraph_chunks

    chunks = [chunk for chunk in chunks if chunk != '.']

    return chunks


def get_pages(pdf_path: str) -> str: 

    loader = PyPDFLoader(pdf_path)
    
    seiten_docs = loader.load()
    
    paragraph_splitter = CharacterTextSplitter(
        separator="\n\n",  
        chunk_size=4_096,  
        chunk_overlap=0    
    )
    
    paragraph_docs = paragraph_splitter.split_documents(seiten_docs)
    
    paragraphs = [re.sub(r"[^a-zA-ZäöüÄÖÜß.\s]", "", doc.page_content).lower().strip() for doc in paragraph_docs]
    return paragraphs


def get_sentences(text: str) -> str: 
    sentences = text.split(".")
    sentences = [sentence.lower().strip() for sentence in sentences]

    return sentences
