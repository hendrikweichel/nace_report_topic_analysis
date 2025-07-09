import pandas as pd
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

### Helper

alphabet = "abcdefghijklmnopqrstuvwxyz"

def cosine_similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))


###

def create_sentence_nace_code_similarities(chunks: str) -> pd.DataFrame:
    
    df = pd.read_csv("NACE_Rev2_Structure_Explanatory_Notes_EN__1_.tsv", sep="\t")

    df_first_level = df[df["ID"].apply(lambda x: not x.isnumeric())]
    df_first_level = df_first_level.dropna(subset=["Includes"])
    df_first_level.reset_index(drop=True, inplace=True)
    df_first_level["IncludesAlso"] = df_first_level["IncludesAlso"].fillna("")
    labels = [df_first_level.loc[i, "Includes"] + " " + df_first_level.loc[i, "IncludesAlso"] for i in range(len(df_first_level))]
    df_first_level["Labels"] = labels


    embedding_model = HuggingFaceEmbeddings(
                        #model_name="sentence-transformers/all-MiniLM-L12-v2",
                        model_name="sentence-transformers/all-mpnet-base-v2",
                        #model_kwargs=model_kwargs,
                        #encode_kwargs=encode_kwargs
                    )
    
    ### Embed sentences
    df_sentences = pd.DataFrame(chunks, columns=["Sentences"])
    df_sentences["Embeddings"] = embedding_model.embed_documents(df_sentences["Sentences"].to_list())



    ### Embed classes
    df_first_level["Embeddings"] = embedding_model.embed_documents(df_first_level["Labels"].to_list())

    ### Caclulate Similarities
    for i, row in df_first_level.iterrows(): 
        similarities = df_sentences["Embeddings"].apply(lambda x: cosine_similarity(x, row.Embeddings))
        df_sentences[f"Scores_{alphabet[i]}_{row.NAME}"] = similarities
    
    return df_sentences
