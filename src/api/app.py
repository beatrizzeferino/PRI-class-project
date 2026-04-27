import json
import os
from fastapi import FastAPI, Query
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite que o HTML aceda à API
    allow_methods=["*"],
    allow_headers=["*"],
)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_PATH = BASE_DIR / "src" / "frontend"
DATA_PATH = BASE_DIR / "scraper_results.json"
app.mount("/static", StaticFiles(directory=FRONTEND_PATH), name="static")


AVAILABLE_METHODS = ["tfidf_custom", "sklearn", "boolean"]

# --------- LOAD DATA ---------
def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

data = load_data()


# --------- ROOT ---------
@app.get("/")
def read_index():
    file_path = FRONTEND_PATH / "index.html"
    # Retorna o ficheiro index.html quando acedes a http://127.0.0.1:8000
    return FileResponse(str(file_path))

@app.get("/results")
def read_results():
    file_path = FRONTEND_PATH / "results.html"
    return FileResponse(str(file_path))

# --------- MOCK ALGORITHMS (temporário) ---------
def tfidf_custom_search(query: str):
    query_l = query.lower()

    results = [doc for doc in data if query_l in doc.get("title", "").lower() or query_l in doc.get("abstract", "").lower()]
    for i, r in enumerate(results):
        r["score"] = 0.99 - (i * 0.05) # Simula scores decrescentes
    return results[:15] # Retorna os 15 melhores

def sklearn_search(query: str):
    query_l = query.lower()

    results = [doc for doc in data if query_l in doc.get("title", "").lower()]
    for r in results:
        r["score"] = 0.88
    return results[:10]

def boolean_search(query: str):
    query_l = query.lower()
    # Simula lógica booleana (score fixo 1.0)
    results = [doc for doc in data if query_l in doc.get("title", "").lower()]
    for r in results:
        r["score"] = 1.0
    return results

# --------- ALGORITHM ROUTER ---------
METHOD_MAP = {
    "tfidf_custom": tfidf_custom_search,
    "sklearn": sklearn_search,
    "boolean": boolean_search
}

def run_algorithm(query: str, method: str):
    func = METHOD_MAP.get(method)
    if not func:
        return []
    return func(query)

# --------- SEARCH ENDPOINT (O motor principal) ---------
@app.get("/search")
def search(
    query: str = Query(...),
    method: str = Query("tfidf_custom")
):
    query_lower = query.lower()
    
    # 1. Filtra os dados reais do ficheiro JSON (armazenados em 'data')
    # Procura no título ou no resumo
    filtered_results = [
        item for item in data 
        if query_lower in item.get("title", "").lower() 
        or query_lower in item.get("abstract", "").lower()
    ]

    # 2. Atribui scores diferentes baseados no método (Simulação para já)
    results = []
    for doc in filtered_results:
        doc_copy = doc.copy()
        if method == "tfidf_custom":
            doc_copy["score"] = 0.92  # Aqui entrará a lógica do teu tfidf.py
        elif method == "boolean":
            doc_copy["score"] = 1.0
        else:
            doc_copy["score"] = 0.85
        results.append(doc_copy)

    # 3. Retorna os resultados reais
    return {
        "query": query,
        "method": method,
        "results": results[:20]  # Limita aos primeiros 20 para performance
    }
# --------- DOCUMENT ---------
@app.get("/document/{doc_id}")
def get_document(doc_id: int):
    for item in data:
        if item.get("id") == doc_id:
            return item

    return {"error": "Document not found"}


# --------- LIST ALGORITHMS ---------
@app.get("/algorithms")
def get_algorithms():
    return {"algorithms": AVAILABLE_METHODS}