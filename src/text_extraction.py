from typing import List
import re
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

    return text
    
def get_sliding_window(text: str, window_len: int = 3) -> List[str]: 

    sentences = text.split(".")
    sentences = [". ".join(sentences[i:i+window_len+1]) for i in range(len(sentences)-window_len)]
    sentences = [sentence.lower().strip() for sentence in sentences]

    return sentences

def get_paragraphs(pdf_path: str) -> str: 

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
