"""
test_indice_manual.py
Teste manual do IndiceInvertido

Executa:
    python test_indice_manual.py
"""

import json
import os

from src.search.indice import IndiceInvertido


# ══════════════════════════════════════════════════════════════════════════════
# Configuração
# ══════════════════════════════════════════════════════════════════════════════

CAMINHO_CORPUS = "processed_corpus.json"

CAMINHO_INDICE = "tests/indice_invertido.json"

PASTA_TEXTOS = "textos_processados"

TERMO1 = "use"
TERMO2 = "system"

TOP_K = 5


# ══════════════════════════════════════════════════════════════════════════════
# Utilitários
# ══════════════════════════════════════════════════════════════════════════════

def sep(titulo=""):

    print("\n" + "─" * 60)

    if titulo:
        print(f"  {titulo}")
        print("─" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# Testes
# ══════════════════════════════════════════════════════════════════════════════

def executar_testes():

    # ──────────────────────────────────────────────────────────────────
    # 1. Carregar corpus
    # ──────────────────────────────────────────────────────────────────

    sep("1. Carregamento do corpus")

    if not os.path.isfile(CAMINHO_CORPUS):

        print(f"[ERRO] Corpus não encontrado: {CAMINHO_CORPUS}")
        return

    with open(CAMINHO_CORPUS, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    print(f"  Documentos carregados: {len(corpus)}")

    # ──────────────────────────────────────────────────────────────────
    # 2. Construir índice
    # ──────────────────────────────────────────────────────────────────

    sep("2. Construção do índice")

    indice = IndiceInvertido()

    pasta = PASTA_TEXTOS if os.path.isdir(PASTA_TEXTOS) else None

    if pasta:
        print(f"  Pasta de textos encontrada: {pasta}")
    else:
        print(f"  [INFO] Pasta '{PASTA_TEXTOS}' não encontrada.")

    indice.construir(corpus, pasta_textos=pasta)

    # ──────────────────────────────────────────────────────────────────
    # 3. Estatísticas
    # ──────────────────────────────────────────────────────────────────

    sep("3. Estatísticas do índice")

    stats = indice.estatisticas()

    print(f"  Nº documentos   : {stats['num_documentos']}")
    print(f"  Nº termos únicos: {stats['num_termos_unicos']}")

    print("\n  Top termos por DF:")

    for termo, df in stats["top_10_termos_por_df"]:

        print(f"    {termo:<20} df={df}")

    # ──────────────────────────────────────────────────────────────────
    # 4. Posting lists
    # ──────────────────────────────────────────────────────────────────

    sep(f"4. Posting Lists: '{TERMO1}' e '{TERMO2}'")

    pl1 = indice.obter_posting_list(TERMO1)
    pl2 = indice.obter_posting_list(TERMO2)

    if pl1:

        print(f"\n  Termo '{TERMO1}'")

        print(f"    DF: {pl1.df}")
        print(f"    Skip pointers: {pl1.skip_pointers}")

        for p in pl1.postings[:TOP_K]:

            print(
                f"    - doc_id={p['doc_id']} "
                f"(tf={p['tf']})"
            )

    else:
        print(f"  [INFO] Termo '{TERMO1}' não encontrado.")

    if pl2:

        print(f"\n  Termo '{TERMO2}'")

        print(f"    DF: {pl2.df}")
        print(f"    Skip pointers: {pl2.skip_pointers}")

        for p in pl2.postings[:TOP_K]:

            print(
                f"    - doc_id={p['doc_id']} "
                f"(tf={p['tf']})"
            )

    else:
        print(f"  [INFO] Termo '{TERMO2}' não encontrado.")

    # ──────────────────────────────────────────────────────────────────
    # 5. Interseção com skip pointers
    # ──────────────────────────────────────────────────────────────────

    sep(f"5. Interseção AND: '{TERMO1}' AND '{TERMO2}'")

    if pl1 and pl2:

        resultado = indice.intersetar_com_skip(pl1, pl2)

        print(
            f"  {len(resultado.postings)} "
            f"documento(s) em comum"
        )

        for p in resultado.postings[:TOP_K]:

            meta = indice.documentos.get(p["doc_id"], {})

            titulo = meta.get("titulo", p["doc_id"])

            print(
                f"    - {titulo[:70]}"
            )

            print(
                f"      doc_id={p['doc_id']}  "
                f"tf_total={p['tf']}"
            )

    else:

        print("  [INFO] Não foi possível fazer interseção.")

    # ──────────────────────────────────────────────────────────────────
    # 6. Persistência
    # ──────────────────────────────────────────────────────────────────

    sep("6. Persistência")

    os.makedirs(os.path.dirname(CAMINHO_INDICE), exist_ok=True)

    indice.guardar(CAMINHO_INDICE)

    indice2 = IndiceInvertido()

    indice2.carregar(CAMINHO_INDICE)

    assert (
        indice2.num_documentos ==
        indice.num_documentos
    ), "Falha no número de documentos"

    assert (
        len(indice2.indice) ==
        len(indice.indice)
    ), "Falha no número de termos"

    print("  Índice recarregado com sucesso.")

    # ──────────────────────────────────────────────────────────────────
    # 7. Verificação dos skip pointers reconstruídos
    # ──────────────────────────────────────────────────────────────────

    sep("7. Verificação dos skip pointers")

    termo_teste = TERMO1

    pl_original = indice.obter_posting_list(termo_teste)

    pl_recarregada = indice2.obter_posting_list(termo_teste)

    if pl_original and pl_recarregada:

        print(
            f"  Skip pointers originais : "
            f"{pl_original.skip_pointers}"
        )

        print(
            f"  Skip pointers carregados: "
            f"{pl_recarregada.skip_pointers}"
        )

        assert (
            pl_original.skip_pointers ==
            pl_recarregada.skip_pointers
        )

        print("  Reconstrução validada.")

    # ──────────────────────────────────────────────────────────────────
    # Conclusão
    # ──────────────────────────────────────────────────────────────────

    sep("CONCLUÍDO")

    print("Todos os testes passaram com sucesso.")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    executar_testes()