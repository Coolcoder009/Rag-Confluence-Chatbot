import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

# Path for storing FAISS DB
FAISS_DB_PATH = "Data/vectorstores/db_faiss"
os.makedirs("Data/vectorstores", exist_ok=True)

# Load HuggingFace Embeddings
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_model = get_embeddings()

# Load JSON data
def load_confluence_data():
    json_path = "confluence_data.json"  # Update with actual path
    if not os.path.exists(json_path):
        print("❌ No JSON data found!")
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Convert content to embeddings and store in FAISS
def store_in_faiss():
    data = load_confluence_data()
    if not data:
        return

    # Prepare text data from JSON
    text_data = [value for value in data.values()]
    titles = [key for key in data.keys()]

    # Generate embeddings
    print("Generating embeddings...")
    text_embeddings = []
    for text in tqdm(text_data):
        embedding = embedding_model.embed_query(text)
        text_embeddings.append((text, embedding))  # Store as tuple (text, embedding)

    # Create FAISS index and save it
    db = FAISS.from_embeddings(text_embeddings, embedding_model)
    db.save_local(FAISS_DB_PATH)

    print("✅ FAISS storage completed!")

store_in_faiss()
