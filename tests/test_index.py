import sys
import os
import json

# Ajustar o path para encontrar a pasta 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from search.corpusProcessor import Indexer

def setup_temp_dataset(filename):
    """Cria um mini dataset com idiomas mistos para o teste."""
    dados = [
        {
            "doi": "10.PT.001",
            "language": "por",
            "title": "Sistemas de Informação",
            "abstract": "Os investigadores estão a estudar novos algoritmos.",
            "authors": ["Silva, J."],
            "year": "2024"
        },
        {
            "doi": "10.EN.002",
            "language": "en",
            "title": "Information Systems",
            "abstract": "Researchers are studying new algorithms.",
            "authors": ["Smith, A."],
            "year": "2024"
        }
    ]
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=4)

def test_all_combinations():
    caminho_json = "temp_test_data.json"
    setup_temp_dataset(caminho_json)
    
    # Inicializa o Indexer apenas uma vez (para carregar modelos)
    print("⏳ Carregando Indexer e Modelos NLP...")
    idx = Indexer()
    
    # Definição das matrizes de teste (todas as possibilidades)
    opcoes_sw = [True, False]
    opcoes_norm = ['lemma', 'stem', None]

    print(f"\n{'='*70}")
    print(f"{'MODO':<25} | {'IDIOMA':<10} | {'AMOSTRA DE TOKENS'}")
    print(f"{'='*70}")

    for sw in opcoes_sw:
        for norm in opcoes_norm:
            label = f"SW:{sw} | Norm:{norm}"
            
            # EXECUTAR INDEXAÇÃO COM ESTA CONFIGURAÇÃO
            resultados = idx.processar_dataset(
                caminho_json, 
                remove_stopwords=sw, 
                normalization_method=norm
            )

            # Verificar resultados para PT e EN
            for doi, doc in resultados.items():
                idioma = doc['idioma']
                tokens = doc['tokens_pesquisa']
                # Mostrar apenas os primeiros 4 tokens para não encher o terminal
                amostra = ", ".join(tokens[:5])
                print(f"{label:<25} | {idioma:<10} | [{amostra}...]")

    # Limpeza
    if os.path.exists(caminho_json):
        os.remove(caminho_json)
    
    print(f"{'='*70}")
    print("✅ TESTE DE TODAS AS POSSIBILIDADES CONCLUÍDO!")

if __name__ == "__main__":
    test_all_combinations()