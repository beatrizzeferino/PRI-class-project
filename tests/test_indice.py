"""
test_indice_manual.py — Teste manual do IndiceInvertido
Executa com:  python test_indice_manual.py
"""

import os
import json
from src.search.indice import IndiceInvertido


# ── Configuração ────────────────────────────────────────────────────────────
CAMINHO_CORPUS  = "processed_corpus.json"
CAMINHO_INDICE  = "tests/indice_invertido.json"
PASTA_TEXTOS      = "textos_processados"   
TERMO1, TERMO2  = "use", "system"
QUERY_TFIDF     = "machine learning neural network"
QUERY_BOOLEANA  = "machine AND learning"
QUERY_OR        = "neural OR graph"
QUERY_NOT       = "NOT python"
QUERY_COMPLEXA  = "(neural OR deep) AND learning"
AUTOR_PESQUISA  = "Santos"
TOP_K           = 5
# ────────────────────────────────────────────────────────────────────────────


def sep(titulo=""):
    print(f"\n{'─'*55}")
    if titulo:
        print(f"  {titulo}")
        print(f"{'─'*55}")


def executar_testes():
    # ── 1. Carregar corpus ──────────────────────────────────────────────────
    if not os.path.exists(CAMINHO_CORPUS):
        print(f"[Erro] Ficheiro '{CAMINHO_CORPUS}' não encontrado.")
        return

    sep("1. Carregamento do corpus")
    with open(CAMINHO_CORPUS, encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"  Documentos carregados: {len(corpus)}")

    # ── 2. Construir índice ─────────────────────────────────────────────────
    sep("2. Construção do índice")
    indice = IndiceInvertido()
    indice.construir(corpus)

    # Usa o método compatível com processed_corpus + textos opcionais
    pasta = PASTA_TEXTOS if os.path.isdir(PASTA_TEXTOS) else None
    if pasta:
        print(f"  Pasta de textos encontrada: {pasta}")
    else:
        print(f"  [Info] Pasta '{PASTA_TEXTOS}' não encontrada — só metadados.")

    indice.construir(corpus, pasta_textos=pasta)

    # ── 3. Estatísticas ─────────────────────────────────────────────────────
    sep("3. Estatísticas do índice")
    stats = indice.estatisticas()
    print(f"  Documentos    : {stats['num_documentos']}")
    print(f"  Termos únicos : {stats['num_termos_unicos']}")
    print(f"  Top 10 termos por DF:")
    for termo, df in stats["top_10_termos_por_df"]:
        print(f"    {termo:<20} df={df}")

    # ── 4. Interseção AND com skip pointers ─────────────────────────────────
    sep(f"4. AND com skip pointers: '{TERMO1}' AND '{TERMO2}'")
    pl1 = indice.obter_posting_list(TERMO1)
    pl2 = indice.obter_posting_list(TERMO2)

    if pl1 and pl2:
        resultado = indice.intersetar_com_skip(pl1, pl2)
        print(f"  {len(resultado.postings)} documento(s) em comum:")
        for p in resultado.postings[:TOP_K]:
            titulo = indice.documentos.get(p["doc_id"], {}).get("titulo", p["doc_id"])
            print(f"    - {titulo[:70]}  (tf={p['tf']})")
    else:
        falta = TERMO1 if not pl1 else TERMO2
        print(f"  [Aviso] Termo '{falta}' não encontrado no índice.")

    # ── 5. Pesquisas booleanas ───────────────────────────────────────────────
    sep("5. Pesquisas booleanas")
    for q in [QUERY_BOOLEANA, QUERY_OR, QUERY_NOT, QUERY_COMPLEXA]:
        ids = indice.pesquisa_booleana(q)
        print(f"  [{q}]  →  {len(ids)} resultado(s)")
        for doc_id in ids[:3]:
            titulo = indice.documentos.get(doc_id, {}).get("titulo", doc_id)
            print(f"      • {titulo[:70]}")

    # ── 6. Ranking TF-IDF ───────────────────────────────────────────────────
    sep(f"6. Ranking TF-IDF: '{QUERY_TFIDF}'")
    resultados_tfidf = indice.pesquisa_tfidf(QUERY_TFIDF, top_k=TOP_K)
    if resultados_tfidf:
        for r in resultados_tfidf:
            print(f"  {r['score']:.4f}  {r['titulo'][:65]}")
            print(f"           Autores: {', '.join(r['autores'][:2])}")
    else:
        print("  [Info] Nenhum resultado.")

    # ── 8. Persistência ─────────────────────────────────────────────────────
    sep("8. Guardar e recarregar índice")
    os.makedirs(os.path.dirname(CAMINHO_INDICE), exist_ok=True)
    indice.guardar(CAMINHO_INDICE)

    indice2 = IndiceInvertido()
    indice2.carregar(CAMINHO_INDICE)
    assert indice2.num_documentos == indice.num_documentos, "Falha: num_documentos diferente após recarregar!"
    assert len(indice2.indice) == len(indice.indice), "Falha: nº de termos diferente após recarregar!"
    print("  Índice recarregado e validado com sucesso.")

    sep("CONCLUÍDO — todos os testes manuais passaram")


if __name__ == "__main__":
    executar_testes()