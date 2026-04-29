import json
import os
from fastapi import FastAPI, Query
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from src.search.booleano import modeloBooleano
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

modelo_bool = modeloBooleano()
modelo_bool.construir_matriz(str(DATA_PATH))
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


def boolean_search(query: str) -> List[Dict]:
    """Lógica exclusiva para o modelo Booleano"""
    if not query.strip():
        return []
    
    try:
        # Chama a função avaliar_query do ficheiro booleano.py
        resultado_binario = modelo_bool.avaliar_query(query)
        
        # Converte o vetor de bits nos documentos reais do JSON
        results = []
        for i, bit in enumerate(resultado_binario):
            if bit == 1:
                doc = data[i].copy()
                doc["score"] = 1.0  # Score binário
                results.append(doc)
        return results
    except Exception as e:
        print(f"Erro no motor booleano: {e}")
        return []

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
    method: str = Query("tfidf_custom"),
    year_min: int = Query(1950),
    year_max: int = Query(2026)
):
    search_func = METHOD_MAP.get(method, tfidf_custom_search)
    base_results = search_func(query)
    
    final_results = []
    for doc in base_results:
        raw_year = doc.get("year", "0")
        doc_year = 0

        if raw_year:
            # 1. Se for lista, pega o primeiro elemento
            if isinstance(raw_year, list) and len(raw_year) > 0:
                raw_year = str(raw_year[0])
            else:
                raw_year = str(raw_year)

            # 2. Limpeza: Manter apenas os números
            digits_only = "".join(filter(str.isdigit, raw_year))
            
            # 3. CORREÇÃO CRUCIAL: Se tivermos uma data completa (ex: 20251125),
            # pegamos apenas nos primeiros 4 dígitos para ter o ano.
            if len(digits_only) >= 4:
                doc_year = int(digits_only[:4])
            elif digits_only:
                doc_year = int(digits_only)

        # 4. Filtro com o ano já corrigido
        if doc_year == 0 or (year_min <= doc_year <= year_max):
            final_results.append(doc)
        else:
            print(f"DEBUG: Cortado - Ano {doc_year} fora do range {year_min}-{year_max}")

    return {
        "query": query,
        "method": method,
        "results": final_results[:50]
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