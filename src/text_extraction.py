from typing import List
import re
import pandas as pd
import pymupdf
from langchain.document_loaders import PyPDFLoader 
from langchain.text_splitter import CharacterTextSplitter

def get_text_from_pdf(pdf_path: str) -> str: 
    loader = PyPDFLoader(pdf_path)
    seiten_docs = loader.load()
    text = " ".join([page.page_content for page in seiten_docs])

    return text

def preprocess_text(text: str) -> str: 
    text = text.replace(" -", "")
    text = text.replace("\n", "")
    text = re.sub(r"[^a-zA-ZäöüÄÖÜß.\s]", "", text)
    text = re.sub(r' +', ' ', text)

    return text
    
def get_sliding_window(text: str, window_len: int = 3) -> List[str]: 

    sentences = text.split(".")
    sentences = [". ".join(sentences[i:i+window_len+1]) for i in range(len(sentences)-window_len)]
    sentences = [sentence.lower().strip() for sentence in sentences]

    return sentences

def get_paragraphs(pdf_path: str) -> str: 

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
