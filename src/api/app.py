import json
import os
from fastapi import FastAPI, Query
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Imports das classes locais
from src.search.booleano import ModeloBooleano
from src.search.corpusProcessor import CorpusProcessor

app = FastAPI()

# Configuração de CORS para permitir comunicação com o Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURAÇÃO DE CAMINHOS ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_PATH = BASE_DIR / "src" / "frontend"
DATA_PATH = BASE_DIR / "scraper_results.json"

# Montar ficheiros estáticos (CSS, JS)
app.mount("/static", StaticFiles(directory=FRONTEND_PATH), name="static")

# --------- CARREGAMENTO DE DADOS ---------
def load_data():
    if not DATA_PATH.exists():
        print(f"ERRO: Ficheiro {DATA_PATH} não encontrado!")
        return []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# 1. Carregar dados brutos primeiro
data = load_data()

# 2. Criar mapeamento DOI -> Documento para recuperação rápida
db_documentos = {doc.get('doi'): doc for doc in data if doc.get('doi')}

# --------- INICIALIZAÇÃO DOS MOTORES DE BUSCA ---------

# 3. Processar o corpus para o modelo booleano[cite: 8, 12]
processor = CorpusProcessor()
corpus_dict = processor.processar_dataset(str(DATA_PATH))

# 4. Instanciar e construir a matriz do Modelo Booleano
modelo_bool = ModeloBooleano(
    corpus_processado=corpus_dict,
    remove_stopwords=True,
    normalization_method='lemma',
    language='english'
)
modelo_bool.construir_matriz(str(DATA_PATH))

AVAILABLE_METHODS = ["tfidf_custom", "sklearn", "boolean"]

# --------- ROTAS DE NAVEGAÇÃO ---------
@app.get("/")
def read_index():
    return FileResponse(str(FRONTEND_PATH / "index.html"))

@app.get("/results")
def read_results():
    return FileResponse(str(FRONTEND_PATH / "results.html"))

# --------- FUNÇÕES DE PESQUISA ---------

def tfidf_custom_search(query: str):
    query_l = query.lower()
    results = [doc for doc in data if query_l in doc.get("title", "").lower() or query_l in doc.get("abstract", "").lower()]
    for i, r in enumerate(results):
        r["score"] = 0.99 - (i * 0.001)
    return results

def sklearn_search(query: str):
    query_l = query.lower()
    results = [doc for doc in data if query_l in doc.get("title", "").lower()]
    for r in results:
        r["score"] = 0.88
    return results

def boolean_search(query: str) -> List[Dict]:
    """Lógica para o modelo Booleano usando DOIs"""
    if not query.strip():
        return []
    
    try:
        # executar_pesquisa devolve uma lista de DOIs[cite: 8]
        dois_encontrados = modelo_bool.executar_pesquisa(query)
        
        # Converter DOIs nos documentos completos usando o mapeamento db_documentos
        results = []
        for doi in dois_encontrados:
            if doi in db_documentos:
                doc = db_documentos[doi].copy()
                doc["score"] = 1.0  # Relevância binária[cite: 12]
                results.append(doc)
        return results
    except Exception as e:
        print(f"Erro no motor booleano: {e}")
        return []

# --------- ROUTER DE ALGORITMOS ---------
METHOD_MAP = {
    "tfidf_custom": tfidf_custom_search,
    "sklearn": sklearn_search,
    "boolean": boolean_search
}

def run_algorithm(query: str, method: str):
    func = METHOD_MAP.get(method)
    # Se não encontrar o método ou for inválido, usa o booleano como padrão
    if not func:
        return boolean_search(query)
    return func(query)

# --------- ENDPOINT PRINCIPAL DE PESQUISA ---------
@app.get("/search")
def search(
    query: str = Query(...),
    method: str = Query("boolean"),
    year_min: int = Query(1950),
    year_max: int = Query(2026)
):
    # 1. Obter resultados brutos do algoritmo selecionado[cite: 12]
    base_results = run_algorithm(query, method)
    
    final_results = []
    
    # 2. Processar metadados e aplicar filtros de ano[cite: 12]
    for doc in base_results:
        raw_year = doc.get("year", "0")
        doc_year = 0

        if raw_year:
            if isinstance(raw_year, list) and len(raw_year) > 0:
                raw_year = str(raw_year[0])
            else:
                raw_year = str(raw_year)

            # Limpar string para obter apenas dígitos (ex: "2023-10" -> "202310")
            digits_only = "".join(filter(str.isdigit, raw_year))
            
            # Extrair o ano (primeiros 4 dígitos)[cite: 12]
            if len(digits_only) >= 4:
                doc_year = int(digits_only[:4])
            elif digits_only:
                doc_year = int(digits_only)

        # 3. Filtragem por intervalo de anos[cite: 12]
        if doc_year == 0 or (year_min <= doc_year <= year_max):
            final_results.append(doc)

    return {
        "query": query,
        "method": method,
        "results": final_results[:50] # Limitar a 50 para performance
    }

# --------- OUTROS ENDPOINTS ---------
@app.get("/document/{doi}")
def get_document(doi: str):
    doc = db_documentos.get(doi)
    if doc:
        return doc
    return {"error": "Document not found"}

@app.get("/algorithms")
def get_algorithms():
    return {"algorithms": AVAILABLE_METHODS}
