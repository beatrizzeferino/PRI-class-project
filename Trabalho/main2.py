import scraper2

def main():
    # URL da coleção (Exemplo: Engenharia Biomédica ou similar)
    collection_id = "1822/14397"
    base_url = f"https://repositorium.uminho.pt/handle/{collection_id}"

    # max_items=200 para a tua necessidade real
    # output_file define onde guardar incrementalmente
    scraper_instance = scraper2.UMinhoDSpace8Scraper(
        base_url, 
        max_items=50, 
        output_file='scraper_results2.json'
    )
    
    print("Iniciando o scraper...")
    final_results = scraper_instance.scrape()
    
    print(f"\nTarefa concluída!")
    print(f"Novos documentos extraídos nesta sessão: {len(final_results)}")
    print("Consulta o ficheiro 'scraper_results2.json' para ver o progresso total.")

if __name__ == "__main__":
    main()