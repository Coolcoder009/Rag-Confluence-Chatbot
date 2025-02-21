"""This script uploads the json data into vector db"""
import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm


FAISS_DB_PATH = "Data/vectorstores/db_faiss"
os.makedirs("Data/vectorstores", exist_ok=True)


def get_embeddings():
    """This function returns the embedding model"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_model = get_embeddings()


def load_confluence_data():
    """This function loads the data from json"""
    json_path = "confluence_data.json"  # Update with actual path
    if not os.path.exists(json_path):
        print("No JSON data found!")
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def store_in_faiss():
    """This function uploads the data into db"""
    data = load_confluence_data()
    if not data:
        return


    text_data = [value for value in data.values()]

    print("Generating embeddings...")
    text_embeddings = []
    for text in tqdm(text_data):
        embedding = embedding_model.embed_query(text)
        text_embeddings.append((text, embedding))  # Store as tuple (text, embedding)


    db = FAISS.from_embeddings(text_embeddings, embedding_model)
    db.save_local(FAISS_DB_PATH)

    print("FAISS storage completed!")

store_in_faiss()
