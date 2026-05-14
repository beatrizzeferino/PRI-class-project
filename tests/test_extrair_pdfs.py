

import unittest
from unittest.mock import patch, MagicMock
import json
import os
import sys
import shutil

# Garante que o Python consegue encontrar a pasta 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.scraper.extrair_pdfs import PDFExtractor

class TestPDFExtractor(unittest.TestCase):
    def setUp(self):
        """Prepara o ambiente antes de cada teste."""
        self.ficheiro_teste = 'teste_corpus_pdf.json'
        self.pasta_saida = 'teste_textos_pdfs'
        
        # Criamos um corpus falso com 3 cenários diferentes
        self.dados_falsos = {
            "doi_1": {
                "title": "Artigo Verdadeiro",
                "link": "http://site.com/real.pdf",
                "has_pdf_txt": False
            },
            "doi_2": {
                "title": "Artigo Paywall",
                "link": "http://site.com/falso.html",
                "has_pdf_txt": False
            },
            "doi_3": {
                "title": "Artigo Já Processado",
                "link": "http://site.com/outro.pdf",
                "has_pdf_txt": True # Este já está extraído!
            }
        }
        
        with open(self.ficheiro_teste, 'w', encoding='utf-8') as f:
            json.dump(self.dados_falsos, f)

    def tearDown(self):
        """Limpa o ambiente depois dos testes terminarem."""
        if os.path.exists(self.ficheiro_teste):
            os.remove(self.ficheiro_teste)
        
        # Apaga a pasta de testes e tudo o que estiver lá dentro
        if os.path.exists(self.pasta_saida):
            shutil.rmtree(self.pasta_saida)

    def test_01_cria_pasta_destino(self):
        """Testa se a classe cria a pasta de destino ao ser instanciada."""
        PDFExtractor(output_dir=self.pasta_saida)
        self.assertTrue(os.path.exists(self.pasta_saida), "A pasta de saída não foi criada.")

    # O @patch substitui as funções reais pelas nossas "falsas" durante este teste
    @patch('src.scraper.extrair_pdfs.subprocess.run')
    @patch('src.scraper.extrair_pdfs.requests.get')
    def test_02_ignora_falso_pdf(self, mock_requests, mock_subprocess):
        """Testa se o bloqueio de Magic Bytes (%PDF) está a funcionar."""
        
        # Configuramos para devolver HTML em vez de PDF
        resposta_falsa = MagicMock()
        resposta_falsa.content = b'<!DOCTYPE html><html>Erro 403</html>'
        mock_requests.return_value = resposta_falsa

        extrator = PDFExtractor(output_dir=self.pasta_saida)
        extrator.extrair_pdfs(self.ficheiro_teste, self.ficheiro_teste, limite=5)

        # Ler o ficheiro atualizado
        with open(self.ficheiro_teste, 'r', encoding='utf-8') as f:
            resultado = json.load(f)

        # O doi_2 era o HTML. O nosso código deve ter detetado e marcado como False
        self.assertFalse(resultado["doi_2"]["has_pdf_txt"])
        
        # O subprocess (pdftotext) NUNCA deve ter sido chamado porque bloqueámos a tempo
        mock_subprocess.assert_not_called()

    @patch('src.scraper.extrair_pdfs.subprocess.run')
    @patch('src.scraper.extrair_pdfs.requests.get')
    def test_03_sucesso_extracao(self, mock_requests, mock_subprocess):
        """Testa o caminho feliz: Descarrega um PDF real e atualiza o JSON."""
        
        # Configuramos para devolver os Magic Bytes corretos do PDF
        resposta_falsa = MagicMock()
        resposta_falsa.content = b'%PDF-1.4 Dados Falsos do PDF'
        mock_requests.return_value = resposta_falsa

        extrator = PDFExtractor(output_dir=self.pasta_saida)
        extrator.extrair_pdfs(self.ficheiro_teste, self.ficheiro_teste, limite=5)

        with open(self.ficheiro_teste, 'r', encoding='utf-8') as f:
            resultado = json.load(f)

        # O doi_1 era válido e não estava processado. Agora deve estar True
        self.assertTrue(resultado["doi_1"]["has_pdf_txt"])
        self.assertIn("pdf_txt_path", resultado["doi_1"])
        
        # Verifica se o subprocess (pdftotext) foi chamado pelo menos uma vez
        self.assertTrue(mock_subprocess.called)

        # O doi_3 já estava processado desde o início, não deve ter sido alterado
        self.assertTrue(resultado["doi_3"]["has_pdf_txt"])

if __name__ == '__main__':
    unittest.main(verbosity=2)