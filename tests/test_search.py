import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), r'C:\Users\carol\Desktop\Informática Médica\2º semestre\Processamento e Recuperação de Informação\Aulas práticas\Projeto\PRI-class-project')))

from src.search.booleano import modeloBooleano
import json

def teste_booleano():

    test_data = [
            {"title": "Gato preto", "abstract": "Um gato de cor preta.", "authors": ["Joao"], "year": "2020"},
            {"title": "Cão branco", "abstract": "Um cão de cor branca.", "authors": ["Maria"], "year": "2021"},
            {"title": "Gato e Cão", "abstract": "Uma história sobre um gato e um cão.", "authors": ["Joao", "Maria"], "year": "2022"}
        ]
        
    with open("test_docs.json", "w") as f:
        json.dump(test_data, f)

    modelo = modeloBooleano()
    modelo.construir_matriz("test_docs.json")

    testes = [
            ("Simples", "gato"),                      # Deve trazer Doc 1 e 3
            ("AND Explícito", "gato AND cão"),        # Deve trazer apenas Doc 3
            ("AND Implícito", "gato cão"),  # Deve trazer apenas Doc 3
            ("OR", "preto OR branco"),                # Deve trazer Doc 1 e 2
            ("NOT", "gato NOT cão"),                  # Deve trazer apenas Doc 1
            ("Precedência", "gato AND (preto OR cão)"),# Deve trazer Doc 1 e 3
            ("Campos Extra", "Joao 2022")             # Testar se procura em autores e anos
        ]

    print("\n--- RESULTADOS DOS TESTES ---")
    for nome, query in testes:
        resultado = modelo.executar_pesquisa(query)
        print(f"{nome} [{query}]: {resultado}")

if __name__ == "__main__":
    teste_booleano()