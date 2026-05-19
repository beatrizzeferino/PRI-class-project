import json
import os
import tempfile
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
DATA_PATH = BASE_DIR / "processed_corpus.json"
TOKENS_PDF_PATH = BASE_DIR / "textos_processados"

# Montar ficheiros estáticos (CSS, JS)
app.mount("/static", StaticFiles(directory=FRONTEND_PATH), name="static")

# --------- CARREGAMENTO DE DADOS ---------
def load_data():
    if not DATA_PATH.exists():
        print(f"ERRO: Ficheiro {DATA_PATH} não encontrado!")
        return []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # processed_corpus.json é um dict {doi: doc} — converter para lista
    if isinstance(raw, dict):
        return list(raw.values())
    return raw

# 1. Carregar dados brutos primeiro (lista de dicts)
data = load_data()

# 2. Criar mapeamento DOI -> Documento para recuperação rápida
db_documentos = {doc.get('doi'): doc for doc in data if doc.get('doi')}

# 2b. Criar mapeamento TÍTULO -> Documento para o modelo booleano
db_por_titulo = {doc.get('titulo', '').strip(): doc for doc in data if doc.get('titulo')}

# --------- INICIALIZAÇÃO DOS MOTORES DE BUSCA ---------

# 3. Processar o corpus — o CorpusProcessor lê o ficheiro diretamente e espera
#    uma lista, por isso usamos um ficheiro temporário com os dados convertidos.
processor = CorpusProcessor()
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp:
    json.dump(data, tmp, ensure_ascii=False)
    tmp_path = tmp.name

corpus_dict = processor.processar_dataset(tmp_path)
os.remove(tmp_path)

# 4. Modelo Booleano
modelo_bool = ModeloBooleano(
    corpus_processado=corpus_dict,
    pasta_tokens_pdf=str(TOKENS_PDF_PATH),
    remove_stopwords=True,
    normalization_method='lemma',
    language='english'
)
modelo_bool.construir_matriz()

# 5. Índice Invertido — necessário para o TFIDF personalizado
try:
    from src.search.indice import IndiceInvertido
    indice = IndiceInvertido()
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
        pasta_tokens_pdf=str(TOKENS_PDF_PATH),
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

# --------- FUNÇÕES AUXILIARES ---------
def _enrich_results(ranking, max_results=50):
    """
    Recebe uma lista de (doc_id, score) e devolve os documentos enriquecidos
    com os metadados disponíveis e o score.
    Os campos do corpus processado são: titulo, abstrato, autores, ano, doi, link.
    Mapeamos para os nomes que o frontend espera: title, abstract, authors, year, doi, link.
    """
    results = []
    for doc_id, score in ranking[:max_results]:
        meta = db_documentos.get(str(doc_id), {})
        enriched = dict(meta) if meta else {"doi": doc_id}
        enriched["score"] = round(float(score), 4)

        # Mapear campos do corpus (português) para nomes usados no frontend (inglês)
        enriched["title"]    = enriched.get("titulo") or doc_id
        enriched["abstract"] = enriched.get("abstrato") or ""
        enriched["authors"]  = enriched.get("autores") or ""
        enriched["year"]     = enriched.get("ano") or ""
        enriched["link"]     = enriched.get("link") or enriched.get("url") or ""
        enriched.setdefault("doi", doc_id)

        results.append(enriched)
    return results

# --------- FUNÇÕES DE PESQUISA ---------

def tfidf_custom_search(query: str, remove_sw: bool, norm: str, tf_w: str, idf_w: str):
    """TF-IDF implementado de raiz com suporte a diferentes esquemas de peso."""
    if modelo_tfidf_custom is None:
        print("[ERRO] TF-IDF personalizado indisponível.")
        return []

    modelo_tfidf_custom.remove_stopwords = remove_sw
    modelo_tfidf_custom.normalization_method = norm
    modelo_tfidf_custom.tf_scheme = tf_w
    modelo_tfidf_custom.idf_scheme = idf_w

    ranking = modelo_tfidf_custom.rank_documentos(query)
    return _enrich_results(ranking)


def tfidf_sklearn_search(query: str, remove_sw: bool, norm: str):
    """TF-IDF com scikit-learn."""
    modelo_tfidf_sklearn.remove_stopwords = remove_sw
    modelo_tfidf_sklearn.normalization_method = norm

    ranking = modelo_tfidf_sklearn.rank_documentos(query)
    return _enrich_results(ranking)


def boolean_search(query: str, remove_sw: bool, norm: str):
    """Pesquisa booleana. O ModeloBooleano devolve doc_ids (DOIs)."""
    modelo_bool.remove_stopwords = remove_sw
    modelo_bool.normalization_method = norm

    doc_ids = modelo_bool.executar_pesquisa(query)

    # Booleano não tem score contínuo — usamos 1.0 para todos
    ranking = [(doc_id, 1.0) for doc_id in doc_ids]
    return _enrich_results(ranking)


# --------- ROUTER DE ALGORITMOS ---------
def run_algorithm(query: str, method: str, remove_sw: bool, norm: str, tf_w: str, idf_w: str):
    if method == "boolean":
        return boolean_search(query, remove_sw, norm)
    elif method == "tfidf_custom":
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
    norm_method = 'stem' if stemming else ('lemma' if lematizacao else None)

    base_results = run_algorithm(query, method, stopwords, norm_method, tf_weighting, idf_weighting)

    final_results = []

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

        if doc_year == 0 or (year_min <= doc_year <= year_max):
            final_results.append(doc)

    print(f"[SEARCH] Query: '{query}' | Resultados após filtro: {len(final_results)}")

    return {
        "query": query,
        "method": method,
        "results": final_results[:50]
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