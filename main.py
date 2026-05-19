import json
import time
from src.scraper.scraper import UMinhoDSpace8Scraper
from src.search.corpusProcessor import CorpusProcessor
from src.search.booleano import ModeloBooleano
from src.search.indice import IndiceInvertido
from src.search.tfidf import TFIDF, TFIDF_Sklearn
from src.search.processorPdfs import ProcessorPdfs
from src.scraper.extrair_pdfs import PDFExtractor

def load_config(config_path="config.json"):
    """Lê as configurações do ficheiro JSON."""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Erro: O ficheiro '{config_path}' não foi encontrado.")
        return None
    
def main():
    config = load_config()
    url=f"{config['repo_url']}/{config['collection']}"
    if config:
        print(f"A extrair URL: {url}")
        print(f"Limite de artigos: {config['max_items']}")

        start_time = time.time()

    # Create an instance of the Scraper class
    # The scraper will automatically detect Chrome in default locations
    scraper_instance = UMinhoDSpace8Scraper(
        base_url = url,
        max_items= config['max_items'],
        output_file= config['output_file'])
    
    final_results = scraper_instance.scrape()

    # --- 2. FASE DE CONFIGURAÇÃO (enquanto ainda nao esta integrado com interface) ---
    print(f"\n{'='*20} CONFIGURAÇÃO {'='*20}")
    
    remover_sw = input("Remover Stop Words? (s/n): ").lower() == 's'
    
    print("\nMétodo de Normalização:")
    print("1. Lematização (Mais preciso)")
    print("2. Stemming (Mais rápido)")
    print("3. Nenhum (Mantém palavras originais)")
    escolha_norm = input("Escolha (1/2/3): ")
    
    mapping = {'1': 'lemma', '2': 'stem', '3': None}
    metodo_norm = mapping.get(escolha_norm, 'lemma')

    
    print(f"\n{'='*20} 3. INDEXAÇÃO E PROCESSAMENTO NLP {'='*20}")
    
    processador = CorpusProcessor() # O Indexer já carrega o TextProcessor internamente
    # Processamos o dataset com as escolhas feitas acima
    documentos_indexados = processador.processar_dataset(
        config['output_file'],
        remove_stopwords=remover_sw,
        normalization_method=metodo_norm,
        caminho_saida="processed_corpus.json"
    )

    extrator = PDFExtractor(output_dir="textos_pdfs")
    extrator.extrair_pdfs(
        corpus_file="processed_corpus.json", 
        output_file="processed_corpus.json", # Atualiza o JSON com caminhos dos TXT brutos
        limite=20
    )

    processorPdfs = ProcessorPdfs(processed_dir="textos_processados")
    documentos_indexados = processorPdfs.processar_e_guardar_tokens(
        caminho_corpus_base="processed_corpus.json",
        remove_stopwords=remover_sw,
        normalization_method=metodo_norm
    )

    print(f"\n{'='*20} 🔍 RESULTADO DO PROCESSAMENTO {'='*20}")
    
    # dar print para confirmar
    amostra_ids = list(documentos_indexados.keys())[:2]
    
    for i, doc_id in enumerate(amostra_ids, 1):
        info = documentos_indexados[doc_id]
        print(f"\n📄 Documento #{i} | DOI: {doc_id}")
        print(f"   📌 Título: {info['titulo']}")
        print(f"   🌍 Idioma Detetado: {info['idioma'].upper()}")
        print(f"   🔢 Total de Tokens: {len(info['tokens_pesquisa'])}")
        
        # Mostra apenas os primeiros 15 tokens para conferir a "limpeza"
        print(f"   ✨ Amostra de Tokens (limpos): {info['tokens_pesquisa'][:15]}...")
    print(f"\n{'='*60}")
    print(f"✅ Processamento concluído com sucesso!")

    with open("processed_corpus.json", "r", encoding="utf-8") as f:
        corpus = json.load(f)

    #------- Indice Invertido --------------
    indice= IndiceInvertido()
    indice.construir(documentos_indexados, pasta_textos="textos_processados")
    print("Índice construído com sucesso!")
    print(f"Número de documentos: {indice.num_documentos}")
    print(f"Número de termos únicos: {len(indice.indice)}")

    #------- MODELO BOOLEANO ---------------
    modelo_booleano= ModeloBooleano(
        corpus_processado=corpus,
        pasta_tokens_pdf="textos_processados",
        remove_stopwords=remover_sw,
        normalization_method=metodo_norm,
        language="english" #assume o modelo ingles no processamento das querys
    )
    modelo_booleano.construir_matriz()
    
    '''
    Testar modelo booleano

    print(f"\n{'='*20} BOOLEAN SEARCH MODE {'='*20}")

    while True:
        query = input("\nQuery Boolean (ou 'exit'): ")
        
        if query.lower() == "exit":
            break

        resultados = modelo_booleano.executar_pesquisa(query)

        print(f"\n Resultados ({len(resultados)} documentos):")
        for r in resultados[:10]:
            print(" -", r)
    '''

    #TF-IDF:
    print("\nEscolha o modelo Tf-Idf:")
    print("1- Implementação manual")
    print("2- Sklearn")

    escolha= input("Opção: ")
    
    if escolha== "1":
        print("\n================ TF SCHEMES ================")
        print("O TF (Term Frequency) define como a frequência de um termo no documento influencia o peso.")
        print("Escolhe como queres medir essa importância:\n")

        print("1 - Raw TF")
        print("    -> Usa diretamente o número de vezes que o termo aparece.")
        print("    -> Simples, mas favorece documentos longos.")

        print("\n2 - Log TF")
        print("    -> Usa 1 + log(tf).")
        print("    -> Reduz o impacto de termos muito repetidos.")

        print("\n3 - Augmented TF")
        print("    -> Normaliza a frequência entre documentos.")
        print("    -> Evita que documentos longos dominem o ranking.\n")

        tf_choice = input("Escolha TF (1/2/3): ")


        print("\n================ IDF SCHEMES ================")
        print("O IDF mede o quão raro um termo é no corpus.")
        print("Termos raros têm maior peso.\n")

        print("1 - Standard IDF")
        print("    -> log(N / df)")
        print("    -> Penaliza termos muito comuns.")

        print("\n2 - Binary IDF")
        print("    -> 1 se tf > 0, caso contrário 0")
        print("    -> Considera apenas se o termo existe no documento, ignorando repetições.")

        print("\n3 - Smooth IDF")
        print("    -> log(1 + N / df)")
        print("    -> Versão mais estável numericamente.")

        print("\n4 - Probabilistic IDF")
        print("    -> log((N - df) / df)")
        print("    -> Dá mais peso a termos realmente discriminativos.\n")

        idf_choice = input("Escolha IDF (1/2/3): ")

        #mapeamento das escolhas
        tf_map = {
            "1": "raw",
            "2": "binary",
            "3": "log",
            "4": "augmented"
        }

        idf_map = {
            "1": "standard",
            "2": "smooth",
            "3": "probabilistic"
        }
        modelo_tfidf= TFIDF(indice,
          documentos=corpus,
          pasta_tokens_pdf="textos_processados",
          tf_scheme=tf_map[tf_choice],
          idf_scheme=idf_map[idf_choice],
          remove_stopwords=remover_sw,
          normalization_method=metodo_norm,
          language="english"  
        )
        matriz= modelo_tfidf.gerar_matriz_similaridade()
        #print(matriz)

    elif escolha =="2":
        modelo_tfidf= TFIDF_Sklearn(
          documentos_processados=documentos_indexados,
          remove_stopwords=remover_sw,
          normalization_method=metodo_norm,
          language="english"  
        )
    else:
        print("Opção inválida.")
        return
    
    '''
    Testar TF-IDF
    '''
    while True:
        query = input("Query: ")

        if query.lower() == "exit":
            break

        resultados = modelo_tfidf.rank_documentos(query)

        for doc_id, score in resultados[:10]:
            print(doc_id, score)

if __name__ == "__main__":
    main()