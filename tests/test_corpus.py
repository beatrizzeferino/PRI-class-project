import unittest
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.search.corpusProcessor import CorpusProcessor

class TestCorpusProcessor(unittest.TestCase):
    def setUp(self):
        """Prepara o ambiente criando um ficheiro JSON falso com casos extremos."""
        self.ficheiro_entrada = 'teste_input_corpus.json'
        self.ficheiro_saida = 'teste_output_corpus.json'

        self.dados_falsos = [
            {
                "doi": "10.123/artigo-original", 
                "title": "Inteligência Artificial", 
                "abstract": "Um resumo sobre IA.",
                "language": "por",
                "year": "2023"
            },
            {
                "doi": "N/A", # DOI não existe
                "title": "Artigo Sem DOI", 
                "abstract": "Teste de falha no scraper.",
                "language": "en",
                "keywords": ["test", "scraper"]
            },
            {
                "doi": "10.123/artigo-original", #DOI Duplicado!
                "title": "Artigo Duplicado", 
                "abstract": "Este tem o mesmo DOI do primeiro.",
                "language": "en"
            }
        ]
        
        with open(self.ficheiro_entrada, 'w', encoding='utf-8') as f:
            json.dump(self.dados_falsos, f)

    def tearDown(self):
        """Limpa o ambiente, apagando os ficheiros de teste que criámos."""
        if os.path.exists(self.ficheiro_entrada):
            os.remove(self.ficheiro_entrada)
        if os.path.exists(self.ficheiro_saida):
            os.remove(self.ficheiro_saida)

    def test_01_ficheiro_nao_encontrado(self):
        """Testa se o sistema reage bem quando o JSON de entrada não existe."""
        processador = CorpusProcessor()
        resultado = processador.processar_dataset('ficheiro_fantasma.json', caminho_saida=self.ficheiro_saida)
        
        self.assertEqual(resultado, {}, "Deve devolver um dicionário vazio se o ficheiro não existir.")

    def test_02_resolucao_de_dois(self):
        """Testa a tua lógica para DOIs N/A e DOIs duplicados."""
        processador = CorpusProcessor()
        docs = processador.processar_dataset(self.ficheiro_entrada, caminho_saida=self.ficheiro_saida)
        
        # O resultado deve ter processado os 3 artigos, sem se sobreporem
        self.assertEqual(len(docs), 3, "Deviam ter sido indexados 3 artigos distintos.")
        
        chaves = list(docs.keys())
        
        # O 1º artigo deve ter o DOI normal
        self.assertIn("10.123/artigo-original", chaves)
        
        # O 2º artigo tinha "N/A", pelo teu código deve ter recebido "doc_0"
        self.assertIn("doc_0", chaves, "O artigo com DOI 'N/A' não gerou o ID esperado.")
        
        # O 3º artigo era duplicado, pelo teu código deve ter recebido "10.123/artigo-original_1"
        self.assertIn("10.123/artigo-original_1", chaves, "O artigo duplicado não recebeu o sufixo esperado.")

    def test_03_preservacao_de_metadados_e_idiomas(self):
        """Testa se o título, idioma e tokens estão a ser guardados no dicionário."""
        processador = CorpusProcessor()
        docs = processador.processar_dataset(self.ficheiro_entrada, caminho_saida=self.ficheiro_saida)
        
        # Vamos verificar os dados do primeiro artigo (em português)
        artigo_pt = docs["10.123/artigo-original"]
        self.assertEqual(artigo_pt["titulo"], "Inteligência Artificial")
        self.assertEqual(artigo_pt["idioma"], "portuguese")
        self.assertIn("tokens_pesquisa", artigo_pt)
        
        # Vamos verificar os dados do artigo em inglês
        artigo_en = docs["doc_0"]
        self.assertEqual(artigo_en["idioma"], "english")

    def test_04_guardar_json(self):
        """Testa se o ficheiro final é realmente criado no disco."""
        processador = CorpusProcessor()
        processador.processar_dataset(self.ficheiro_entrada, caminho_saida=self.ficheiro_saida)
        
        # Verifica se o ficheiro foi criado
        self.assertTrue(os.path.exists(self.ficheiro_saida), "O ficheiro JSON de saída não foi criado.")
        
        # Verifica se o conteúdo do ficheiro corresponde ao que esperamos
        with open(self.ficheiro_saida, 'r', encoding='utf-8') as f:
            dados_guardados = json.load(f)
            
        self.assertEqual(len(dados_guardados), 3)

if __name__ == '__main__':
    unittest.main(verbosity=2)