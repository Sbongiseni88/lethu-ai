from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd
import os

df=pd.read_csv("rag.csv")
embeddings=OllamaEmbeddings(model="mxbai-embed-large")

db_location="./chrome_langchain_db"
add_documents= not os.path.exists(db_location)

if add_documents:
    documents=[]
    ids=[]

    for i,row in df.iterrows():
        
