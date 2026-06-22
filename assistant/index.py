import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Paths relative to the script execution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(BASE_DIR)
CATALOGUE_PATH = os.path.join(WORKSPACE_DIR, "catalogue.json")
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(BASE_DIR, "index_metadata.json")

# Model configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def format_item_for_embedding(item):
    """
    Format a catalogue item into a text string designed for dense semantic retrieval.
    Includes explicit attribute labeling to guide the vector representation.
    """
    # Normalize potential dash characters
    name = item.get("name", "").replace("\u2014", " - ")
    desc = item.get("description", "").replace("\u2014", " - ")
    
    parts = [
        f"Product Name: {name}",
        f"SKU: {item.get('sku', '')}",
        f"Category: {item.get('category', '')}",
        f"Brand: {item.get('brand', '')}",
        f"Vehicle Fitment: {item.get('vehicle_fitment', '')}",
        f"Description: {desc}",
        f"Price: {item.get('price_inr', '')} INR",
        f"Stock Available: {item.get('stock', '')} units"
    ]
    return " | ".join(parts)

def build_index():
    """
    Loads product catalogue, formats text chunks, runs them through SentenceTransformers,
    creates a FAISS index, and saves index files.
    """
    print(f"Loading product catalogue from: {CATALOGUE_PATH}")
    if not os.path.exists(CATALOGUE_PATH):
        raise FileNotFoundError(f"Catalogue file not found at {CATALOGUE_PATH}")

    with open(CATALOGUE_PATH, 'r', encoding='utf-8') as f:
        catalogue = json.load(f)

    print(f"Loaded {len(catalogue)} items. Formatting text chunks...")
    chunks = [format_item_for_embedding(item) for item in catalogue]
    
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    print("Generating dense vector embeddings (this might take a few moments)...")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    
    # L2 normalize embeddings for cosine similarity via Inner Product (IP) index
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    
    print(f"Creating FAISS Inner Product index with dimension {dimension}...")
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # Save the index
    print(f"Saving FAISS index to: {INDEX_PATH}")
    faiss.write_index(index, INDEX_PATH)
    
    # Save catalogue metadata (list of dictionaries ordered matching FAISS index rows)
    # We strip out description and formatting to keep it lightweight, or keep full items for easy grounding
    print(f"Saving metadata mapping to: {METADATA_PATH}")
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(catalogue, f, indent=2, ensure_ascii=False)
        
    print("Indexing completed successfully!")

if __name__ == "__main__":
    build_index()
