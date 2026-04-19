import json
import time
from scraper import UMinhoDSpace8Scraper

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

    if config:
        print(f"A extrair URL: {config['base_url']}")
        print(f"Limite de artigos: {config['max_items']}")

        start_time = time.time()

    # Create an instance of the Scraper class
    # The scraper will automatically detect Chrome in default locations
    scraper_instance = UMinhoDSpace8Scraper(
        base_url = config['base_url'],
        max_items= config['max_items'],
        output_file= config['output_file'])
    
    final_results = scraper_instance.scrape()
    end_time = time.time()
    tempo_total= round((end_time-start_time)/60,2)

    print(f"\n Extração concluída em {tempo_total} minutos.")
    print(f"Total de {len(final_results)} artigos guardados no ficheiro '{config['output_file']}'.")

if __name__ == "__main__":
    main()