import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Paths relative to the script execution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(BASE_DIR, "index_metadata.json")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

class CatalogueRetriever:
    """
    Handles loading the FAISS vector index and metadata, embedding queries,
    and performing semantic retrieval for catalogue products.
    """
    def __init__(self):
        self.model = None
        self.index = None
        self.metadata = None
        self._load_resources()

    def _load_resources(self):
        # 1. Load the SentenceTransformer model
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # 2. Check if FAISS index and metadata files exist
        if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(
                f"Index files not found. Please run index.py first to build the index.\n"
                f"Expected index at: {INDEX_PATH}\n"
                f"Expected metadata at: {METADATA_PATH}"
            )
            
        # 3. Load the FAISS index
        self.index = faiss.read_index(INDEX_PATH)
        
        # 4. Load the metadata mapping
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

    def retrieve(self, query: str, top_k: int = 5) -> list:
        """
        Embeds the query and performs a semantic search over the index.
        Returns a list of dictionaries, each representing a product with a similarity score.
        """
        if not query or query.strip() == "":
            return []
            
        # Embed query and normalize for Inner Product (IP) cosine similarity
        query_vector = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)
        
        # Search index
        scores, indices = self.index.search(query_vector, top_k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            item = self.metadata[idx].copy()
            item['similarity_score'] = float(score)
            results.append(item)
            
        return results

# Self-test block
if __name__ == "__main__":
    retriever = CatalogueRetriever()
    print("Testing retriever with query: 'brake pads for Pulsar'")
    test_results = retriever.retrieve("brake pads for Pulsar", top_k=3)
    for r in test_results:
        print(f"- SKU: {r['sku']}, Name: {r['name']}, Brand: {r['brand']}, Fitment: {r['vehicle_fitment']}, Score: {r['similarity_score']:.4f}")
