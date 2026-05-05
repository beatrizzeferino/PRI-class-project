import os
from src.search.corpusProcessor import CorpusProcessor
# Importa as classes do teu ficheiro original
# Substitui 'indice' pelo nome real do teu ficheiro .py
from src.search.indice import IndiceInvertido 

def executar_testes():
    # 1. Processar documentos com o CorpusProcessor
    indexer = CorpusProcessor()
    
    # Verifica se o ficheiro de dados existe antes de processar
    caminho_dados = "scraper_results.json"
    if not os.path.exists(caminho_dados):
        print(f"[Erro] Ficheiro {caminho_dados} não encontrado.")
        return

    print("--- Processando Dataset ---")
    documentos_processados = indexer.processar_dataset(
        caminho_dados,
        remove_stopwords=True,
        normalization_method='lemma'
    )

    # 2. Construir o índice
    indice = IndiceInvertido()
    #indice.reset_total()
    indice.construir_de_indexer(documentos_processados)

    # 3. Estatísticas
    stats = indice.estatisticas()
    print(f"\nEstatísticas do índice:")
    print(f"  Documentos    : {stats['num_documentos']}")
    print(f"  Termos únicos : {stats['num_termos_unicos']}")
    print(f"  Top 10 termos : {stats['top_10_termos_por_df']}")

    # 4. Exemplo de interseção com skip pointers
    # Alterei para termos comuns em inglês, ajusta conforme o teu dataset
    termo1, termo2 = "use", "system"
    pl1 = indice.obter_posting_list(termo1)
    pl2 = indice.obter_posting_list(termo2)

    if pl1 and pl2:
        resultado = indice.intersetar_com_skip(pl1, pl2)
        print(f"\nResultados '{termo1} AND {termo2}': {len(resultado.postings)} documentos")
        for p in resultado.postings[:5]:
            meta = indice.documentos.get(p["doc_id"], {})
            titulo = meta.get('titulo', p['doc_id'])
            print(f"  - {titulo} (tf={p['tf']})")
    else:
        print(f"\n[Aviso] Um dos termos ('{termo1}' ou '{termo2}') não foi encontrado no índice.")

    # 5. Guardar índice no disco
    indice.guardar("tests/indice_invertido.json")

if __name__ == "__main__":
    executar_testes()