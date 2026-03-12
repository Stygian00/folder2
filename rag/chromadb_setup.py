# ChromaDB Setup for RAG

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os

CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./rag/chroma_db")

def get_chroma_client():
    return chromadb.Client(Settings(persist_directory=CHROMA_DB_DIR))

def initialize_kb():
    client = get_chroma_client()
    kb_dir = "./rag/knowledge_base"
    docs = []
    metadatas = []
    for fname in os.listdir(kb_dir):
        with open(os.path.join(kb_dir, fname), "r") as f:
            docs.append(f.read())
            metadatas.append({"source": fname})
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection("knowledge_base", embedding_function=embedding_fn)
    collection.add(documents=docs, metadatas=metadatas, ids=[str(i) for i in range(len(docs))])
    print("ChromaDB knowledge base initialized.")

def retrieve_context(query, n_results=2):
    """
    Retrieve relevant context from ChromaDB for a given query/email.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection("knowledge_base", embedding_function=embedding_functions.DefaultEmbeddingFunction())
    results = collection.query(query_texts=[query], n_results=n_results)
    docs = results.get('documents', [[]])[0]
    return '\n'.join(docs) if docs else ''

if __name__ == '__main__':
    initialize_kb()
    