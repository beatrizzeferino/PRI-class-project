"""
test_indice_manual.py — Teste manual do IndiceInvertido
Executa com:  python test_indice_manual.py
"""

import os
import json
from src.search.corpusProcessor import CorpusProcessor
from src.search.indice import IndiceInvertido


# ── Configuração ────────────────────────────────────────────────────────────
CAMINHO_SCRAPER   = "scraper_results.json"
CAMINHO_INDICE    = "tests/indice_invertido.json"
PASTA_TEXTOS      = "textos_processados"          # None se não tiveres
TERMO1, TERMO2    = "use", "system"
QUERY_TFIDF       = "machine learning neural network"
QUERY_BOOLEANA    = "machine AND learning"
QUERY_OR          = "neural OR graph"
QUERY_NOT         = "NOT python"
QUERY_COMPLEXA    = "(neural OR deep) AND learning"
AUTOR_PESQUISA    = "Santos"
TOP_K             = 5
# ────────────────────────────────────────────────────────────────────────────


def sep(titulo=""):
    print(f"\n{'─'*55}")
    if titulo:
        print(f"  {titulo}")
        print(f"{'─'*55}")


def executar_testes():
    # ── 1. Verificar ficheiro de dados ──────────────────────────────────────
    if not os.path.exists(CAMINHO_SCRAPER):
        print(f"[Erro] Ficheiro '{CAMINHO_SCRAPER}' não encontrado.")
        return

    # ── 2. Processar corpus com CorpusProcessor ─────────────────────────────
    sep("1. Processamento do corpus")
    indexer = CorpusProcessor()
    documentos_processados = indexer.processar_dataset(
        CAMINHO_SCRAPER,
        remove_stopwords=True,
        normalization_method="lemma",
    )
    print(f"  Documentos processados: {len(documentos_processados)}")

    # ── 3. Construir índice ─────────────────────────────────────────────────
    sep("2. Construção do índice")
    indice = IndiceInvertido()

    # Usa o método compatível com processed_corpus + textos opcionais
    pasta = PASTA_TEXTOS if os.path.isdir(PASTA_TEXTOS) else None
    if pasta:
        print(f"  Pasta de textos encontrada: {pasta}")
    else:
        print(f"  [Info] Pasta '{PASTA_TEXTOS}' não encontrada — só metadados.")

    indice.construir_de_processed_corpus(documentos_processados, pasta_textos=pasta)

    # ── 4. Estatísticas ─────────────────────────────────────────────────────
    sep("3. Estatísticas do índice")
    stats = indice.estatisticas()
    print(f"  Documentos    : {stats['num_documentos']}")
    print(f"  Termos únicos : {stats['num_termos_unicos']}")
    print(f"  Top 10 termos por DF:")
    for termo, df in stats["top_10_termos_por_df"]:
        print(f"    {termo:<20} df={df}")

    # ── 5. Interseção AND com skip pointers ─────────────────────────────────
    sep(f"4. AND com skip pointers: '{TERMO1}' AND '{TERMO2}'")
    pl1 = indice.obter_posting_list(TERMO1)
    pl2 = indice.obter_posting_list(TERMO2)

    if pl1 and pl2:
        resultado = indice.intersetar_com_skip(pl1, pl2)
        print(f"  {len(resultado.postings)} documento(s) em comum:")
        for p in resultado.postings[:TOP_K]:
            meta  = indice.documentos.get(p["doc_id"], {})
            titulo = meta.get("titulo", p["doc_id"])
            print(f"    - {titulo[:70]}  (tf={p['tf']})")
    else:
        falta = TERMO1 if not pl1 else TERMO2
        print(f"  [Aviso] Termo '{falta}' não encontrado no índice.")

    # ── 6. Pesquisas booleanas ───────────────────────────────────────────────
    sep("5. Pesquisas booleanas")
    queries_bool = [
        QUERY_BOOLEANA,
        QUERY_OR,
        QUERY_NOT,
        QUERY_COMPLEXA,
    ]
    for q in queries_bool:
        ids = indice.pesquisa_booleana(q)
        print(f"  [{q}]  →  {len(ids)} resultado(s)")
        for doc_id in ids[:3]:
            titulo = indice.documentos.get(doc_id, {}).get("titulo", doc_id)
            print(f"      • {titulo[:70]}")

    # ── 7. Ranking TF-IDF ───────────────────────────────────────────────────
    sep(f"6. Ranking TF-IDF: '{QUERY_TFIDF}'")
    resultados_tfidf = indice.pesquisa_tfidf(QUERY_TFIDF, top_k=TOP_K)
    if resultados_tfidf:
        for r in resultados_tfidf:
            print(f"  {r['score']:.4f}  {r['titulo'][:65]}")
            print(f"           Autores: {', '.join(r['autores'][:2])}")
    else:
        print("  [Info] Nenhum resultado.")

    # ── 8. Pesquisa por autor ────────────────────────────────────────────────
    sep(f"7. Pesquisa por autor: '{AUTOR_PESQUISA}'")
    resultados_autor = indice.pesquisa_por_autor(AUTOR_PESQUISA)
    if resultados_autor:
        for r in resultados_autor[:TOP_K]:
            print(f"  [{r['ano']}]  {r['titulo'][:65]}")
            print(f"           {', '.join(r['autores'])}")
    else:
        print(f"  [Info] Nenhum documento com autor '{AUTOR_PESQUISA}'.")

    # ── 9. Persistência ─────────────────────────────────────────────────────
    sep("8. Guardar e recarregar índice")
    os.makedirs(os.path.dirname(CAMINHO_INDICE), exist_ok=True)
    indice.guardar(CAMINHO_INDICE)

    indice2 = IndiceInvertido()
    indice2.carregar(CAMINHO_INDICE)
    assert indice2.num_documentos == indice.num_documentos, "Falha: num_documentos diferente após recarregar!"
    assert len(indice2.indice) == len(indice.indice),       "Falha: nº de termos diferente após recarregar!"
    print("  Índice recarregado e validado com sucesso.")

    sep("CONCLUÍDO — todos os testes manuais passaram")


if __name__ == "__main__":
    executar_testes()