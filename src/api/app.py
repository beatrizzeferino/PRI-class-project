import json
import os
import re
from pathlib import Path
from typing import List, Dict

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Imports das classes locais
from src.search.booleano import ModeloBooleano
from src.search.tfidf import TFIDF, TFIDF_Sklearn
from src.search.nlp import TextProcessor
from src.search.indice import IndiceInvertido

app = FastAPI()

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

app.mount("/static", StaticFiles(directory=FRONTEND_PATH), name="static")

# --------- CARREGAMENTO DE DADOS & INSTANCIAÇÃO ---------
def init_database():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Ficheiro {DATA_PATH} não encontrado!")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# Base de dados centralizada do corpus {doi: dados_documento}
db_documentos = init_database()

# Instanciação dos motores com configurações iniciais padrão
modelo_bool = ModeloBooleano(
    corpus_processado=db_documentos,
    pasta_tokens_pdf=str(TOKENS_PDF_PATH),
    remove_stopwords=True,
    normalization_method='lemma',
    language='english'
)
modelo_bool.construir_matriz()

indice = IndiceInvertido()
indice.construir(db_documentos)

modelo_tfidf_custom = TFIDF(
    indice=indice,
    documentos=db_documentos,
    pasta_tokens_pdf=str(TOKENS_PDF_PATH),
    tf_scheme="log",
    idf_scheme="standard",
    remove_stopwords=True,
    normalization_method="lemma",
    language="english"
)

modelo_tfidf_sklearn = TFIDF_Sklearn(
    documentos_processados=db_documentos,
    remove_stopwords=True,
    normalization_method="lemma",
    language="english"
)

AVAILABLE_METHODS = ["tfidf_custom", "tfidf_sklearn", "boolean"]
nlp_processor = TextProcessor()

# --------- FUNÇÕES AUXILIARES DE SUPORTE ---------
def _obter_texto_completo_documento(doc_id: str, info: dict) -> str:
    """Função auxiliar única para ler o texto combinado (Meta + PDF) de um artigo."""
    texto_base = f"{info.get('titulo', '')} {info.get('abstrato', '')}"
    safe_id = str(doc_id).replace("/", "_").replace("\\", "_")
    caminho_pdf = TOKENS_PDF_PATH / f"{safe_id}.txt"
    
    texto_pdf = ""
    if caminho_pdf.exists():
        try:
            with open(caminho_pdf, "r", encoding="utf-8") as f:
                texto_pdf = f.read()
        except Exception:
            pass
    return f"{texto_base} {texto_pdf}"

def _enrich_results(ranking, max_results=50):
    results = []
    for doc_id, score in ranking[:max_results]:
        meta = db_documentos.get(str(doc_id), {})
        enriched = dict(meta) if meta else {"doi": doc_id}
        enriched["score"] = round(float(score), 4)

        # Mapeamento semântico Português -> Inglês para o Frontend
        enriched["title"]    = enriched.get("titulo") or doc_id
        enriched["abstract"] = enriched.get("abstrato") or ""
        enriched["authors"]  = enriched.get("autores") or ""
        enriched["year"]     = enriched.get("ano") or ""
        enriched["link"]     = enriched.get("link") or enriched.get("url") or ""
        enriched.setdefault("doi", doc_id)
        results.append(enriched)
    return results

# --------- MOTORES DE PESQUISA DINÂMICOS (COM CACHE) ---------
_cache_sklearn = None
_cache_custom  = None
_cache_boolean = None

def tfidf_sklearn_search(query: str, remove_sw: bool, norm: str):
    global _cache_sklearn
    config_key = f"sw_{remove_sw}_norm_{norm}"
    
    if _cache_sklearn != config_key:
        print(f"[Sklearn] A reconstruir matriz dinamicamente...")
        novo_corpus = []
        novos_doc_ids = []
        for doc_id, info in modelo_tfidf_sklearn.documentos.items():
            texto_total = f"{info.get('titulo', '')} {info.get('abstrato', '')}"
            tokens = nlp_processor.process_text(texto_total, language="english", remove_stopwords=remove_sw, normalization_method=norm)
            novo_corpus.append(" ".join(tokens))
            novos_doc_ids.append(doc_id)
            
        modelo_tfidf_sklearn.remove_stopwords = remove_sw
        modelo_tfidf_sklearn.normalization_method = norm
        modelo_tfidf_sklearn.doc_ids = novos_doc_ids
        modelo_tfidf_sklearn.matriz_tfidf = modelo_tfidf_sklearn.vectorizer.fit_transform(novo_corpus)
        _cache_sklearn = config_key

    ranking = modelo_tfidf_sklearn.rank_documentos(query)
    return _enrich_results(ranking)


def tfidf_custom_search(query: str, remove_sw: bool, norm: str, tf_w: str, idf_w: str):
    global _cache_custom
    config_key = f"sw_{remove_sw}_norm_{norm}"
    
    if _cache_custom != config_key:
        print(f"[Custom TF-IDF] A reconstruir Índice Invertido dinamicamente...")
        corpus_atualizado = {}
        for doc_id, info in db_documentos.items():
            texto_total = f"{info.get('titulo', '')} {info.get('abstrato', '')}"
            tokens = nlp_processor.process_text(texto_total, language="english", remove_stopwords=remove_sw, normalization_method=norm)
            corpus_atualizado[doc_id] = {**info, "tokens_pesquisa": tokens}
            
        novo_indice = IndiceInvertido()
        novo_indice.construir(corpus_atualizado)
        
        modelo_tfidf_custom.indice = novo_indice
        modelo_tfidf_custom.documentos = corpus_atualizado
        modelo_tfidf_custom.N = novo_indice.num_documentos
        modelo_tfidf_custom.remove_stopwords = remove_sw
        modelo_tfidf_custom.normalization_method = norm
        _cache_custom = config_key

    modelo_tfidf_custom.tf_scheme = tf_w
    modelo_tfidf_custom.idf_scheme = idf_w
    
    ranking = modelo_tfidf_custom.rank_documentos(query)
    return _enrich_results(ranking)


def boolean_search(query: str, remove_sw: bool, norm: str):
    global _cache_boolean
    config_key = f"sw_{remove_sw}_norm_{norm}"
    
    if _cache_boolean != config_key:
        print(f"[Booleano] A intercetar rotinas e recalcular matriz...")
        
        # Monkey Patching usando a nossa função auxiliar global limpa
        def carregar_tokens_dinamico(doc_id, info_doc):
            texto_completo = _obter_texto_completo_documento(doc_id, info_doc)
            return nlp_processor.process_text(texto_completo, language=modelo_bool.language, remove_stopwords=remove_sw, normalization_method=norm)
            
        modelo_bool.carregar_tokens_documento = carregar_tokens_dinamico
        modelo_bool.remove_stopwords = remove_sw
        modelo_bool.normalization_method = norm
        modelo_bool.construir_matriz()
        _cache_boolean = config_key

    # Higienização cirúrgica dos tokens da Query Booleana
    tokens_query = re.findall(r"\(|\)|\bAND\b|\bOR\b|\bNOT\b|\b\w+\b", query)
    query_ajustada_lista = []
    
    for token in tokens_query:
        if token in ["AND", "OR", "NOT", "(", ")"]:
            query_ajustada_lista.append(token)
        else:
            termos_limpos = nlp_processor.process_text(token, language="english", remove_stopwords=remove_sw, normalization_method=norm)
            query_ajustada_lista.append(termos_limpos[0] if termos_limpos else "termo_inexistente_de_salvaguarda")
                
    doc_ids = modelo_bool.executar_pesquisa(" ".join(query_ajustada_lista))
    ranking = [(doc_id, 1.0) for doc_id in doc_ids]
    return _enrich_results(ranking)

# --------- ROTAS DA API ---------
@app.get("/")
def read_index():
    return FileResponse(str(FRONTEND_PATH / "index.html"))

@app.get("/results")
def read_results():
    return FileResponse(str(FRONTEND_PATH / "results.html"))

@app.get("/algorithms")
def get_algorithms():
    return {"algorithms": AVAILABLE_METHODS}

@app.get("/document/{doi}")
def get_document(doi: str):
    doc = db_documentos.get(doi)
    if doc:
        return doc
    return {"error": "Document not found"}

@app.get("/search")
def search(
    query: str = Query(...),
    method: str = Query("tfidf_sklearn"),
    year_min: int = Query(1950),
    year_max: int = Query(2026),
    stemming: str = Query("false"),
    lematizacao: str = Query("true"),
    stopwords: str = Query("true"),
    tf_weighting: str = Query("raw"),
    idf_weighting: str = Query("standard")
):
    # Conversão explícita de parâmetros string do JS para bool
    stemming_bool    = stemming.lower()    == "true"
    lematizacao_bool = lematizacao.lower() == "true"
    stopwords_bool   = stopwords.lower()   == "true"

    norm_method = 'stem' if stemming_bool else ('lemma' if lematizacao_bool else None)

    # Roteamento centralizado do algoritmo selecionado
    if method == "boolean":
        base_results = boolean_search(query, stopwords_bool, norm_method)
    elif method == "tfidf_custom":
        base_results = tfidf_custom_search(query, stopwords_bool, norm_method, tf_weighting, idf_weighting)
    else:
        base_results = tfidf_sklearn_search(query, stopwords_bool, norm_method)

    # Filtragem temporal otimizada e limpa
    final_results = []
    for doc in base_results:
        raw_year = doc.get("year", "0")
        doc_year = 0

        if raw_year:
            if isinstance(raw_year, list) and len(raw_year) > 0:
                raw_year = str(raw_year[0])
            digits_only = "".join(filter(str.isdigit, str(raw_year)))
            if len(digits_only) >= 4:
                doc_year = int(digits_only[:4])

        if doc_year == 0 or (year_min <= doc_year <= year_max):
            final_results.append(doc)

    return {
        "query": query,
        "method": method,
        "results": final_results[:50]
    }