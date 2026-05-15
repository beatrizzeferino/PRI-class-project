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
from src.search.tfidf import TFIDFCustom, TFIDFSklearn
 
app = FastAPI(
    title="Motor de Pesquisa — RepositóriUM",
    description="Motor de IR com modelos Booleano, TF-IDF Custom e TF-IDF sklearn.",
    version="1.0.0"
)
 
# Configuração de CORS
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
 
# Montar ficheiros estáticos apenas se a pasta existir
if FRONTEND_PATH.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_PATH), name="static")
 
# ------------------------------------------------------------------ #
#  CARREGAMENTO DE DADOS                                              #
# ------------------------------------------------------------------ #
 
def load_data():
    if not DATA_PATH.exists():
        print(f"[Aviso] Ficheiro {DATA_PATH} não encontrado. A usar lista vazia.")
        return []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
 
# 1. Dados brutos
data = load_data()
 
# 2. Mapeamento DOI -> documento completo para recuperação rápida
db_documentos: Dict[str, dict] = {}
for doc in data:
    doi = doc.get("doi")
    if doi and doi != "N/A":
        db_documentos[doi] = doc
 
# ------------------------------------------------------------------ #
#  INICIALIZAÇÃO DOS MOTORES                                          #
# ------------------------------------------------------------------ #
 
# Parâmetros globais de NLP (podem ser expostos como parâmetros de query)
NLP_REMOVE_STOPWORDS = True
NLP_NORMALIZATION = "lemma"
NLP_LANGUAGE = "english"
 
# 3. Processar corpus (partilhado por todos os modelos)
processor = CorpusProcessor()
corpus_dict = processor.processar_dataset(
    str(DATA_PATH),
    remove_stopwords=NLP_REMOVE_STOPWORDS,
    normalization_method=NLP_NORMALIZATION
)
 
# 4. Modelo Booleano
modelo_bool = ModeloBooleano(
    corpus_processado=corpus_dict,
    remove_stopwords=NLP_REMOVE_STOPWORDS,
    normalization_method=NLP_NORMALIZATION,
    language=NLP_LANGUAGE
)
# BUGFIX: construir_matriz() não recebe argumentos
modelo_bool.construir_matriz()
 
# 5. TF-IDF Custom
modelo_tfidf_custom = TFIDFCustom(
    remove_stopwords=NLP_REMOVE_STOPWORDS,
    normalization_method=NLP_NORMALIZATION,
    language=NLP_LANGUAGE
)
modelo_tfidf_custom.construir(corpus_dict)
 
# 6. TF-IDF sklearn
try:
    modelo_tfidf_sklearn = TFIDFSklearn(
        remove_stopwords=NLP_REMOVE_STOPWORDS,
        normalization_method=NLP_NORMALIZATION,
        language=NLP_LANGUAGE
    )
    modelo_tfidf_sklearn.construir(corpus_dict)
    SKLEARN_DISPONIVEL = True
except Exception as e:
    print(f"[Aviso] sklearn não disponível: {e}")
    modelo_tfidf_sklearn = None
    SKLEARN_DISPONIVEL = False
 
AVAILABLE_METHODS = ["boolean", "tfidf_custom"] + (["sklearn"] if SKLEARN_DISPONIVEL else [])
 
# ------------------------------------------------------------------ #
#  ROTAS DE NAVEGAÇÃO                                                 #
# ------------------------------------------------------------------ #
 
@app.get("/", include_in_schema=False)
def read_index():
    index_file = FRONTEND_PATH / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Frontend não encontrado. Use /docs para a API."}
 
@app.get("/results", include_in_schema=False)
def read_results():
    results_file = FRONTEND_PATH / "results.html"
    if results_file.exists():
        return FileResponse(str(results_file))
    return {"message": "Frontend não encontrado."}
 
# ------------------------------------------------------------------ #
#  FUNÇÕES DE PESQUISA                                                #
# ------------------------------------------------------------------ #
 
def _doc_id_to_full(doc_id: str, score: float) -> dict | None:
    """
    Converte um doc_id no documento completo (com score).
    Tenta primeiro por DOI direto; caso contrário, percorre o corpus_dict
    para obter os metadados.
    """
    # Tentar via db_documentos (indexed por DOI)
    if doc_id in db_documentos:
        doc = db_documentos[doc_id].copy()
        doc["score"] = score
        return doc
 
    # Fallback: usar metadados do corpus_dict
    if doc_id in corpus_dict:
        info = corpus_dict[doc_id]
        doc = {
            "doi":      doc_id,
            "title":    info.get("titulo", ""),
            "authors":  info.get("autores", []),
            "year":     info.get("ano", ""),
            "url":      info.get("url", ""),
            "abstract": "",
            "score":    score
        }
        return doc
 
    return None
 
 
def boolean_search(query: str) -> List[Dict]:
    """Pesquisa booleana — devolve relevância binária (score = 1.0)."""
    if not query.strip():
        return []
 
    try:
        doc_ids_encontrados = modelo_bool.executar_pesquisa(query)
        results = []
        for doc_id in doc_ids_encontrados:
            doc = _doc_id_to_full(doc_id, score=1.0)
            if doc:
                results.append(doc)
        return results
    except Exception as e:
        print(f"[Erro] Motor booleano: {e}")
        return []
 
 
def tfidf_custom_search(query: str) -> List[Dict]:
    """Pesquisa TF-IDF com implementação própria, ordenada por similaridade do cosseno."""
    if not query.strip():
        return []
 
    try:
        resultados = modelo_tfidf_custom.pesquisar(query)
        docs = []
        for r in resultados:
            doc = _doc_id_to_full(r["doc_id"], score=r["score"])
            if doc:
                docs.append(doc)
        return docs
    except Exception as e:
        print(f"[Erro] TF-IDF Custom: {e}")
        return []
 
 
def sklearn_search(query: str) -> List[Dict]:
    """Pesquisa TF-IDF com scikit-learn, ordenada por similaridade do cosseno."""
    if not query.strip() or modelo_tfidf_sklearn is None:
        return tfidf_custom_search(query)  # fallback para custom se sklearn indisponível
 
    try:
        resultados = modelo_tfidf_sklearn.pesquisar(query)
        docs = []
        for r in resultados:
            doc = _doc_id_to_full(r["doc_id"], score=r["score"])
            if doc:
                docs.append(doc)
        return docs
    except Exception as e:
        print(f"[Erro] TF-IDF sklearn: {e}")
        return []
 
 
# ------------------------------------------------------------------ #
#  ROUTER DE ALGORITMOS                                               #
# ------------------------------------------------------------------ #
 
METHOD_MAP = {
    "boolean":      boolean_search,
    "tfidf_custom": tfidf_custom_search,
    "sklearn":      sklearn_search,
}
 
def run_algorithm(query: str, method: str) -> List[Dict]:
    func = METHOD_MAP.get(method, boolean_search)
    return func(query)
 
# ------------------------------------------------------------------ #
#  ENDPOINT PRINCIPAL DE PESQUISA                                     #
# ------------------------------------------------------------------ #
 
@app.get("/search", summary="Pesquisa de documentos")
def search(
    query: str = Query(..., description="Texto ou expressão booleana a pesquisar"),
    method: str = Query("boolean", description="Algoritmo: boolean | tfidf_custom | sklearn"),
    year_min: int = Query(1950, description="Ano mínimo de publicação"),
    year_max: int = Query(2026, description="Ano máximo de publicação")
):
    """
    Pesquisa documentos no corpus com o algoritmo selecionado.
    Aplica filtro por intervalo de anos e devolve no máximo 50 resultados.
    """
    base_results = run_algorithm(query, method)
 
    final_results = []
    for doc in base_results:
        raw_year = doc.get("year", "0")
        doc_year = 0
 
        if raw_year:
            if isinstance(raw_year, list) and len(raw_year) > 0:
                raw_year = str(raw_year[0])
            else:
                raw_year = str(raw_year)
 
            digits_only = "".join(filter(str.isdigit, raw_year))
 
            if len(digits_only) >= 4:
                doc_year = int(digits_only[:4])
            elif digits_only:
                doc_year = int(digits_only)
 
        if doc_year == 0 or (year_min <= doc_year <= year_max):
            final_results.append(doc)
 
    return {
        "query":   query,
        "method":  method,
        "total":   len(final_results),
        "results": final_results[:50]
    }
 
# ------------------------------------------------------------------ #
#  OUTROS ENDPOINTS                                                   #
# ------------------------------------------------------------------ #
 
@app.get("/document/{doi:path}", summary="Detalhes de um documento por DOI")
def get_document(doi: str):
    """Devolve os metadados completos de um documento pelo seu DOI."""
    doc = db_documentos.get(doi)
    if doc:
        return doc
    return {"error": f"Documento '{doi}' não encontrado."}
 
 
@app.get("/algorithms", summary="Algoritmos disponíveis")
def get_algorithms():
    """Lista os algoritmos de pesquisa disponíveis."""
    return {"algorithms": AVAILABLE_METHODS}
 
 
@app.get("/stats", summary="Estatísticas do índice")
def get_stats():
    """Devolve estatísticas gerais sobre o corpus carregado."""
    return {
        "total_documentos": len(corpus_dict),
        "total_termos_vocab_custom": len(modelo_tfidf_custom.vocab),
        "sklearn_disponivel": SKLEARN_DISPONIVEL,
        "metodos_disponiveis": AVAILABLE_METHODS
    }
 


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
