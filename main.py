import json
import time
from src.scraper.scraper import UMinhoDSpace8Scraper
from src.scraper.extrair_pdfs import add_texto_scraper
from src.search.corpusProcessor import CorpusProcessor
from src.search.booleano import ModeloBooleano

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

    print("\nA iniciar a extração de texto dos pdfs.")
    add_texto_scraper(
        scraper_file=config["output_file"],
        output_file=config["output_file"],
        limite=20
    )
    print("\nExtração de PDFs concluída.")
    
    end_time = time.time()
    tempo_total= round((end_time-start_time)/60,2)

    print(f"\n Extração concluída em {tempo_total} minutos.")
    print(f"Total de {len(final_results)} artigos guardados no ficheiro '{config['output_file']}'.")

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

    #------- MODELO BOOLEANO ---------------
    modelo_booleano= ModeloBooleano(
        corpus_processado=documentos_indexados,
        remove_stopwords=remover_sw,
        normalization_method=metodo_norm,
        language="english" #assume o modelo ingles no processamento das querys
    )
    modelo_booleano.construir_matriz(config['output_file'])
    
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

if __name__ == "__main__":
    main()