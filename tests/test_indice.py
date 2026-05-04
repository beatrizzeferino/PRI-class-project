import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), r'C:\Users\Public\Bia\Bia\Universidade\4º ano\2º semestre\6. Processamento e Recuperação de Informação\Trabalho\PRI-class-project')))

from src.search.indice import IndiceInvertido, PostingList
import json
import os

def construir_indice_teste():
    """Constrói um IndiceInvertido diretamente a partir de dados de teste (sem Indexer nem NLP)."""
    documentos_processados = {
        "doi:001": {
            "tokens_pesquisa": ["gato", "preto", "cor"],
            "titulo": "Gato preto",
            "url": "http://exemplo.com/1",
            "autores": ["Joao"],
            "ano": "2020",
            "idioma": "portuguese"
        },
        "doi:002": {
            "tokens_pesquisa": ["cao", "branco", "cor"],
            "titulo": "Cão branco",
            "url": "http://exemplo.com/2",
            "autores": ["Maria"],
            "ano": "2021",
            "idioma": "portuguese"
        },
        "doi:003": {
            "tokens_pesquisa": ["gato", "cao", "historia", "gato", "cao"],
            "titulo": "Gato e Cão",
            "url": "http://exemplo.com/3",
            "autores": ["Joao", "Maria"],
            "ano": "2022",
            "idioma": "portuguese"
        }
    }
    indice = IndiceInvertido()
    indice.construir_de_indexer(documentos_processados)
    return indice


def titulos_de_postings(indice, postings):
    """Converte uma lista de postings nos títulos dos documentos correspondentes."""
    return [indice.documentos[p["doc_id"]]["titulo"] for p in postings]


# ------------------------------------------------------------------ #
#  Testes                                                              #
# ------------------------------------------------------------------ #

def teste_construcao():
    """Verifica se o índice foi construído com os termos e documentos certos."""
    print("\n--- TESTE: Construção do índice ---")
    indice = construir_indice_teste()

    assert indice.num_documentos == 3, "Devia ter 3 documentos"
    assert "gato" in indice.indice,   "'gato' devia estar no índice"
    assert "cao"  in indice.indice,   "'cao' devia estar no índice"
    assert "cor"  in indice.indice,   "'cor' devia estar no índice"

    pl_gato = indice.obter_posting_list("gato")
    assert pl_gato.df == 2, f"'gato' devia aparecer em 2 docs, aparece em {pl_gato.df}"

    pl_cor = indice.obter_posting_list("cor")
    assert pl_cor.df == 2, f"'cor' devia aparecer em 2 docs, aparece em {pl_cor.df}"

    print("  ✔ Número de documentos correto (3)")
    print("  ✔ Termos presentes no índice")
    print(f"  ✔ DF de 'gato' = {pl_gato.df} (esperado: 2)")
    print(f"  ✔ DF de 'cor'  = {pl_cor.df} (esperado: 2)")


def teste_tf():
    """Verifica se o TF (frequência do termo no documento) está correto."""
    print("\n--- TESTE: TF (Term Frequency) ---")
    indice = construir_indice_teste()

    # "gato" aparece 2x em doi:003 e 1x em doi:001
    pl_gato = indice.obter_posting_list("gato")
    tf_por_doc = {p["doc_id"]: p["tf"] for p in pl_gato.postings}

    assert tf_por_doc["doi:001"] == 1, f"TF de 'gato' em doi:001 devia ser 1, é {tf_por_doc['doi:001']}"
    assert tf_por_doc["doi:003"] == 2, f"TF de 'gato' em doi:003 devia ser 2, é {tf_por_doc['doi:003']}"

    print(f"  ✔ TF de 'gato' em 'Gato preto'  = {tf_por_doc['doi:001']} (esperado: 1)")
    print(f"  ✔ TF de 'gato' em 'Gato e Cão'  = {tf_por_doc['doi:003']} (esperado: 2)")


def teste_posting_list_ordenada():
    """Verifica se as posting lists ficam ordenadas por doc_id após construção."""
    print("\n--- TESTE: Ordenação das posting lists ---")
    indice = construir_indice_teste()

    for termo, pl in indice.indice.items():
        ids = [p["doc_id"] for p in pl.postings]
        assert ids == sorted(ids), f"Posting list de '{termo}' não está ordenada: {ids}"

    print("  ✔ Todas as posting lists estão ordenadas por doc_id")


def teste_skip_pointers():
    """Verifica se os skip pointers são construídos quando há postings suficientes."""
    print("\n--- TESTE: Skip pointers ---")

    # Criar uma posting list grande o suficiente (> 3 entradas) para ter skip pointers
    pl = PostingList()
    for i in range(16):
        pl.adicionar_posting(f"doc_{i:02d}", i + 1)
    pl.ordenar()
    pl.construir_skip_pointers()

    assert len(pl.skip_pointers) > 0, "Devia ter skip pointers para 16 postings"
    salto_esperado = int(16 ** 0.5)  # sqrt(16) = 4
    assert pl.skip_pointers[0] == salto_esperado, \
        f"Primeiro skip devia apontar para {salto_esperado}, aponta para {pl.skip_pointers[0]}"

    print(f"  ✔ Skip pointers criados para 16 postings (salto = {salto_esperado})")
    print(f"  ✔ skip_pointers[0] = {pl.skip_pointers[0]} (esperado: {salto_esperado})")

    # Com 3 ou menos postings não deve haver skip pointers
    pl_pequena = PostingList()
    for i in range(3):
        pl_pequena.adicionar_posting(f"doc_{i}", 1)
    pl_pequena.ordenar()
    pl_pequena.construir_skip_pointers()
    assert pl_pequena.skip_pointers == {}, "Posting list com <= 3 entradas não devia ter skip pointers"
    print("  ✔ Posting list com ≤ 3 entradas não tem skip pointers")


def teste_intersecao_and():
    """
    Verifica a interseção AND com skip pointers.
    'gato' está em doi:001 e doi:003
    'cao'  está em doi:002 e doi:003
    AND → apenas doi:003
    """
    print("\n--- TESTE: Interseção AND (com skip pointers) ---")
    indice = construir_indice_teste()

    pl_gato = indice.obter_posting_list("gato")
    pl_cao  = indice.obter_posting_list("cao")

    resultado = indice.intersetar_com_skip(pl_gato, pl_cao)
    titulos = titulos_de_postings(indice, resultado.postings)

    assert len(resultado.postings) == 1,       f"AND devia devolver 1 doc, devolveu {len(resultado.postings)}"
    assert "doi:003" == resultado.postings[0]["doc_id"], "O doc AND devia ser doi:003"
    # TF combinado = tf_gato(2) + tf_cao(2) = 4
    assert resultado.postings[0]["tf"] == 4,   f"TF combinado devia ser 4, é {resultado.postings[0]['tf']}"

    print(f"  ✔ 'gato AND cao' → {titulos} (esperado: ['Gato e Cão'])")
    print(f"  ✔ TF combinado = {resultado.postings[0]['tf']} (esperado: 4)")


def teste_uniao_or():
    """
    Verifica a união OR.
    'preto' está em doi:001
    'branco' está em doi:002
    OR → doi:001 e doi:002
    """
    print("\n--- TESTE: União OR ---")
    indice = construir_indice_teste()

    pl_preto  = indice.obter_posting_list("preto")
    pl_branco = indice.obter_posting_list("branco")

    resultado = indice.unir(pl_preto, pl_branco)
    titulos = titulos_de_postings(indice, resultado.postings)

    assert len(resultado.postings) == 2, f"OR devia devolver 2 docs, devolveu {len(resultado.postings)}"
    doc_ids = {p["doc_id"] for p in resultado.postings}
    assert "doi:001" in doc_ids and "doi:002" in doc_ids, f"Docs errados: {doc_ids}"

    print(f"  ✔ 'preto OR branco' → {titulos} (esperado: ['Gato preto', 'Cão branco'])")


def teste_termo_inexistente():
    """Pesquisar um termo que não existe deve devolver None."""
    print("\n--- TESTE: Termo inexistente ---")
    indice = construir_indice_teste()

    resultado = indice.obter_posting_list("unicornio")
    assert resultado is None, "Termo inexistente devia devolver None"

    print("  ✔ 'unicornio' → None (termo não está no índice)")


def teste_persistencia():
    """Verifica que guardar e carregar o índice produz o mesmo resultado."""
    print("\n--- TESTE: Persistência (guardar/carregar) ---")
    caminho = "test_indice_temp.json"

    indice_original = construir_indice_teste()
    indice_original.guardar(caminho)

    indice_carregado = IndiceInvertido()
    indice_carregado.carregar(caminho)

    assert indice_carregado.num_documentos == indice_original.num_documentos, \
        "num_documentos diferente após carregar"
    assert set(indice_carregado.indice.keys()) == set(indice_original.indice.keys()), \
        "Termos no índice diferentes após carregar"
    assert indice_carregado.documentos == indice_original.documentos, \
        "Metadados dos documentos diferentes após carregar"

    pl_orig     = indice_original.obter_posting_list("gato")
    pl_carregada = indice_carregado.obter_posting_list("gato")
    assert pl_carregada.postings == pl_orig.postings, \
        "Postings de 'gato' diferentes após carregar"

    os.remove(caminho)
    print(f"  ✔ Índice guardado e carregado com sucesso ({indice_carregado.num_documentos} docs, {len(indice_carregado.indice)} termos)")
    print(f"  ✔ Metadados dos documentos preservados")
    print(f"  ✔ Postings de 'gato' preservados")


def teste_atualizacao_incremental():
    """
    Verifica que adicionar_documentos() funciona sem reconstruir o índice.
    Adiciona um novo documento com o termo 'papagaio' e verifica que aparece no índice.
    """
    print("\n--- TESTE: Atualização incremental ---")
    indice = construir_indice_teste()

    assert indice.obter_posting_list("papagaio") is None, \
        "'papagaio' não devia existir antes da atualização"

    novos_docs = {
        "doi:004": {
            "tokens_pesquisa": ["papagaio", "verde", "gato"],
            "titulo": "Papagaio verde",
            "url": "http://exemplo.com/4",
            "autores": ["Ana"],
            "ano": "2023",
            "idioma": "portuguese"
        }
    }
    indice.adicionar_documentos(novos_docs)

    assert indice.num_documentos == 4, f"Devia ter 4 docs após atualização, tem {indice.num_documentos}"

    pl_papagaio = indice.obter_posting_list("papagaio")
    assert pl_papagaio is not None,  "'papagaio' devia existir após atualização"
    assert pl_papagaio.df == 1,      f"DF de 'papagaio' devia ser 1, é {pl_papagaio.df}"

    # 'gato' agora deve aparecer em 3 documentos (doi:001, doi:003, doi:004)
    pl_gato = indice.obter_posting_list("gato")
    assert pl_gato.df == 3, f"DF de 'gato' devia ser 3 após atualização, é {pl_gato.df}"

    # Tentar adicionar um doc já existente não deve duplicar
    indice.adicionar_documentos({"doi:004": novos_docs["doi:004"]})
    assert indice.num_documentos == 4, "Documento duplicado não devia aumentar o contador"

    print(f"  ✔ Novo documento adicionado (total: {indice.num_documentos} docs)")
    print(f"  ✔ Termo 'papagaio' aparece no índice (df={pl_papagaio.df})")
    print(f"  ✔ DF de 'gato' atualizado para {pl_gato.df} (esperado: 3)")
    print(f"  ✔ Documento duplicado ignorado corretamente")


# ------------------------------------------------------------------ #
#  Runner                                                              #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    testes = [
        teste_construcao,
        teste_tf,
        teste_posting_list_ordenada,
        teste_skip_pointers,
        teste_intersecao_and,
        teste_uniao_or,
        teste_termo_inexistente,
        teste_persistencia,
        teste_atualizacao_incremental,
    ]

    passou = 0
    falhou = 0

    print("=" * 50)
    print("  TESTES — IndiceInvertido")
    print("=" * 50)

    for teste in testes:
        try:
            teste()
            passou += 1
        except AssertionError as e:
            print(f"  ✘ FALHOU: {e}")
            falhou += 1
        except Exception as e:
            print(f"  ✘ ERRO INESPERADO: {e}")
            falhou += 1

    print("\n" + "=" * 50)
    print(f"  Resultado: {passou} passou(aram) | {falhou} falhou(aram)")
    print("=" * 50)