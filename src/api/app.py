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
from src.search.tfidf import TFIDF, TFIDF_Sklearn
from src.search.corpusProcessor import CorpusProcessor

# Se o teu índice invertido estiver numa classe separada, importa aqui também
# from src.search.indice import IndiceInvertido

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

# 2b. Criar mapeamento TÍTULO -> Documento para o modelo booleano
# (executar_pesquisa devolve títulos, não DOIs)
db_por_titulo = {doc.get('title', '').strip(): doc for doc in data if doc.get('title')}

# --------- INICIALIZAÇÃO DOS MOTORES DE BUSCA ---------

# 3. Processar o corpus — partilhado por todos os modelos
processor = CorpusProcessor()
corpus_dict = processor.processar_dataset(str(DATA_PATH))

# 4. Modelo Booleano
modelo_bool = ModeloBooleano(
    corpus_processado=corpus_dict,
    remove_stopwords=True,
    normalization_method='lemma',
    language='english'
)
modelo_bool.construir_matriz()

# 5. Índice Invertido — necessário para o TFIDF personalizado
#    Ajusta o import/classe conforme o teu projeto
#    Se o CorpusProcessor já devolver o índice, usa-o diretamente.
#    Exemplo genérico:
#
#    indice = IndiceInvertido()
#    indice.construir(corpus_dict)
#
#    Por agora guardamos None e tratamos o caso em run_algorithm.
try:
    from src.search.indice import IndiceInvertido
    indice = IndiceInvertido()
    
    # CORREÇÃO: O método correto no teu indice.py é 'construir_de_indexer'
    indice.construir_de_indexer(corpus_dict)
    
    _indice_disponivel = True
    print("[OK] Índice invertido construído com sucesso.")
except Exception as e:
    print(f"[AVISO] Erro ao carregar índice: {e}")
    indice = None
    _indice_disponivel = False

# 6. TF-IDF personalizado (usa o índice invertido)
if _indice_disponivel:
    modelo_tfidf_custom = TFIDF(
        indice=indice,
        documentos=corpus_dict,
        tf_scheme="log",
        idf_scheme="standard",
        remove_stopwords=True,
        normalization_method="lemma",
        language="english"
    )
else:
    modelo_tfidf_custom = None

# 7. TF-IDF Sklearn
modelo_tfidf_sklearn = TFIDF_Sklearn(
    documentos_processados=corpus_dict,
    remove_stopwords=True,
    normalization_method="lemma",
    language="english"
)

AVAILABLE_METHODS = ["tfidf_custom", "tfidf_sklearn", "boolean"]

# --------- ROTAS DE NAVEGAÇÃO ---------
@app.get("/")
def read_index():
    return FileResponse(str(FRONTEND_PATH / "index.html"))

@app.get("/results")
def read_results():
    return FileResponse(str(FRONTEND_PATH / "results.html"))

# --------- CACHE PARA MODELOS ---------
_matriz_cache = {}

# --------- FUNÇÕES DE PESQUISA ---------

def _enrich_results(ranking, max_results=50):
    """Auxiliar para buscar o conteúdo completo do documento e adicionar o score."""
    results = []
    for doc_id, score in ranking[:max_results]:
        # Tenta encontrar por DOI ou por Título (caso o motor use títulos como IDs)
        doc = db_documentos.get(doc_id) or db_por_titulo.get(str(doc_id).strip())
        if doc:
            enriched = dict(doc)
            enriched["score"] = round(float(score), 4)
            results.append(enriched)
    return results


def tfidf_custom_search(query: str, remove_sw: bool, norm: str, tf_w: str, idf_w: str):
    """TF-IDF implementado de raiz com suporte a diferentes esquemas de peso."""
    if modelo_tfidf_custom is None:
        print("[ERRO] TF-IDF personalizado indisponível.")
        return []

    # 1. Sincronizar as preferências do utilizador com o modelo
    modelo_tfidf_custom.remove_stopwords = remove_sw
    modelo_tfidf_custom.normalization_method = norm
    modelo_tfidf_custom.tf_scheme = tf_w    # "raw", "log", "binary", etc.
    modelo_tfidf_custom.idf_scheme = idf_w  # "idf", "none", etc.

    # 2. Executar o ranking
    ranking = modelo_tfidf_custom.rank_documentos(query)
    return _enrich_results(ranking)


def tfidf_sklearn_search(query: str, remove_sw: bool, norm: str):
    """TF-IDF com scikit-learn (normalmente usa pesos padrão)."""
    # Sincronizar parâmetros de processamento
    modelo_tfidf_sklearn.remove_stopwords = remove_sw
    modelo_tfidf_sklearn.normalization_method = norm

    ranking = modelo_tfidf_sklearn.rank_documentos(query)
    return _enrich_results(ranking)


def boolean_search(query: str, remove_sw: bool = True, norm: str = None):
    """Modelo booleano com suporte a AND, OR, NOT e AND implícito."""
    cache_key = (remove_sw, norm)

    if cache_key not in _matriz_cache:
        modelo_bool.remove_stopwords = remove_sw
        modelo_bool.normalization_method = norm
        modelo_bool.construir_matriz()
        _matriz_cache[cache_key] = (
            modelo_bool.matriz[:],
            modelo_bool.termos_unicos[:],
            modelo_bool.doc_ids[:],
            dict(modelo_bool.termo_indice)
        )
    else:
        matriz, termos, ids, indice_bool = _matriz_cache[cache_key]
        modelo_bool.matriz = [row[:] for row in matriz]
        modelo_bool.termos_unicos = termos[:]
        modelo_bool.doc_ids = ids[:]
        modelo_bool.termo_indice = dict(indice_bool)
        modelo_bool.remove_stopwords = remove_sw
        modelo_bool.normalization_method = norm

    titulos_res = modelo_bool.executar_pesquisa(query)

    results = []
    for titulo in titulos_res:
        doc_completo = db_por_titulo.get(titulo.strip())
        if doc_completo:
            enriched = dict(doc_completo)
            enriched["score"] = 1.0   # booleano não tem score contínuo
            results.append(enriched)

    return results


# --------- ROUTER DE ALGORITMOS ---------
def run_algorithm(query: str, method: str, remove_sw: bool, norm: str, tf_w: str, idf_w: str):
    if method == "boolean":
        return boolean_search(query, remove_sw, norm)
    elif method == "tfidf_custom":
        # Passa os novos parâmetros tf_w e idf_w
        return tfidf_custom_search(query, remove_sw, norm, tf_w, idf_w)
    elif method == "tfidf_sklearn":
        return tfidf_sklearn_search(query, remove_sw, norm)
    return []


# --------- ENDPOINT PRINCIPAL DE PESQUISA ---------
@app.get("/search")
def search(
    query: str = Query(...),
    method: str = Query("tfidf_sklearn"),
    year_min: int = Query(1950),
    year_max: int = Query(2026),
    stemming: bool = Query(False),
    lematizacao: bool = Query(True),
    stopwords: bool = Query(True),
    tf_weighting: str = Query("raw"),
    idf_weighting: str = Query("idf")
):
    # Definir método de normalização
    norm_method = 'stem' if stemming else ('lemma' if lematizacao else None)
    
    # 1. Executar o algoritmo selecionado
    base_results = run_algorithm(query, method, stopwords, norm_method, tf_weighting, idf_weighting)
    
    final_results = []

    # 2. FILTRAGEM POR ANO (Recuperada e corrigida)
    for doc in base_results:
        raw_year = doc.get("year", "0")
        doc_year = 0

        # Lógica de limpeza do ano (trata listas, strings com meses, etc.)
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

        # Só adiciona se estiver dentro do intervalo escolhido no slider
        if doc_year == 0 or (year_min <= doc_year <= year_max):
            final_results.append(doc)

    print(f"[SEARCH] Query: '{query}' | Resultados após filtro: {len(final_results)}")

    return {
        "query": query,
        "method": method,
        "results": final_results[:50]  # Limite para performance
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
