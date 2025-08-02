import os
import faiss
import numpy as np
import pickle
from fastapi import FastAPI, APIRouter
from sentence_transformers import SentenceTransformer
from typing import Dict, List
from fastapi.staticfiles import StaticFiles

app = FastAPI()
api = APIRouter()

MODEL_NAME = "all-MiniLM-L6-v2"
BASE_DATA_DIR = "./data"
INDEX_DIR = "./faiss_indexes"

NAME_TYPES = ["fname", "lname", "full_name"]
model = SentenceTransformer(MODEL_NAME)
dimension = model.get_sentence_embedding_dimension()

# Structure to hold multiple indexes
indexes: Dict[str, Dict[str, faiss.Index]] = {}  # e.g., {"fname": {"hindu_fnames": index, ...}, ...}
embeddings: Dict[str, Dict[str, List[Dict]]] = {}

def load_text_or_csv(file_path: str) -> List[str]:
    ext = os.path.splitext(file_path)[1].lower()
    names = []
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
    elif ext == ".csv":
        import csv
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                for val in row:
                    val = val.strip()
                    if val:
                        names.append(val)
    return names

def build_index_for_folder(name_type: str):
    folder = os.path.join(BASE_DATA_DIR, f"{name_type}_data")
    os.makedirs(folder, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)

    indexes[name_type] = {}
    embeddings[name_type] = {}

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        category_name = os.path.splitext(filename)[0]
        names = load_text_or_csv(file_path)

        index = faiss.IndexFlatL2(dimension)
        index_embeddings = []

        for name in names:
            emb = model.encode(name)
            index.add(np.array([emb], dtype=np.float32))
            index_embeddings.append({"name": name, "embedding": emb})

        # Save index
        faiss.write_index(index, f"{INDEX_DIR}/{name_type}_{category_name}.index")
        with open(f"{INDEX_DIR}/{name_type}_{category_name}.pkl", "wb") as f:
            pickle.dump(index_embeddings, f)

        indexes[name_type][category_name] = index
        embeddings[name_type][category_name] = index_embeddings

def load_all_indexes():
    for name_type in NAME_TYPES:
        build_index_for_folder(name_type)

def search_index(name_type: str, query: str):
    query_embedding = model.encode(query)
    query_embedding = np.array([query_embedding], dtype=np.float32)

    all_results = []
    for category, index in indexes[name_type].items():
        emb_list = embeddings[name_type][category]
        all_doc_embeddings = np.array([e["embedding"] for e in emb_list], dtype=np.float32)
        distances = np.linalg.norm(all_doc_embeddings - query_embedding, axis=1)
        max_distance = max(np.max(distances), 1e-5)
        confidence_scores = 1 - (distances / max_distance)

        max_score = float(np.max(confidence_scores))
        all_results.append({
            "category": category,
            "max_confidence_score": round(max_score, 4)
        })

    return sorted(all_results, key=lambda x: -x["max_confidence_score"])

@api.get("/search_fname")
async def search_fname(query: str):
    return {
        "query": query,
        "results": search_index("fname", query)
    }

@api.get("/search_lname")
async def search_lname(query: str):
    return {
        "query": query,
        "results": search_index("lname", query)
    }

@api.get("/search_fullname")
async def search_fullname(query: str):
    return {
        "query": query,
        "results": search_index("full_name", query)
    }
    
@api.get("/health")
async def health_check():
    return {"status": "ok"}

@api.post("/refresh_indexes")
async def refresh_indexes():
    load_all_indexes()
    return {"status": "reindexed", "types": NAME_TYPES}

app.include_router(api, prefix="/api")
app.mount("/", StaticFiles(directory="llm_server/static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    load_all_indexes()
    uvicorn.run(app, host="0.0.0.0", port=8000)
