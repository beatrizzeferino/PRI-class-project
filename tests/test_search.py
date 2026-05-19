import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), r'C:\Users\carol\Desktop\Informática Médica\2º semestre\Processamento e Recuperação de Informação\Aulas práticas\Projeto\PRI-class-project')))

from src.search.booleano import ModeloBooleano
import json
corpus_teste = {
    "doc1": {
        "tokens_pesquisa": ["cancer", "treatment", "drug"],
        "titulo": "Cancer treatment with drug"
    },
    "doc2": {
        "tokens_pesquisa": ["cancer", "diagnosis"],
        "titulo": "Cancer diagnosis methods"
    },
    "doc3": {
        "tokens_pesquisa": ["heart", "disease"],
        "titulo": "Heart disease study"
    }
}

modelo = ModeloBooleano(
    corpus_processado=corpus_teste,
    remove_stopwords=False,
    normalization_method=None,
    language="english"
)

modelo.construir_matriz()

print(modelo.executar_pesquisa("   cancer"))
print(modelo.executar_pesquisa("cancer treatment"))
print(modelo.executar_pesquisa("cancer NOT treatment"))
print(modelo.executar_pesquisa("(cancer AND diagnosis) OR heart"))
print(modelo.executar_pesquisa("    "))