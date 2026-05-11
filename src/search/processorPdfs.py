import json
import os
import re
from src.search.nlp import TextProcessor

class ProcessorPdfs:
    def __init__(self, processed_dir="textos_processados"):
        """Inicializa a classe, carrega o motor de NLP e e cria a pasta para os textos limpos."""
        self.nlp_processor = TextProcessor()
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

    def _limpar_texto_bruto(self, texto):
        """
        Remove artefactos de PDFs e formatação invisível antes do NLP.
        """
        if not texto:
            return ""

        # Resolver palavras cortadas no fim da linha (ex: "informa-\nção" -> "informação")
        texto_limpo = re.sub(r'(\w+)-\n(\w+)', r'\1\2', texto)

        # Remover (\f), novas linhas (\n, \r) e tabs (\t), trocando por espaços
        texto_limpo = re.sub(r'[\n\f\r\t]', ' ', texto_limpo)

        # Remover caracteres estranhos que às vezes vêm nos PDFs (mantém apenas letras, números e pontuação básica)
        texto_limpo = re.sub(r'[^\w\s.,;:!?()-]', ' ', texto_limpo)

        # Remover múltiplos espaços seguidos
        texto_limpo = re.sub(r'\s+', ' ', texto_limpo)

        return texto_limpo.strip()
    
    def processar_e_guardar_tokens(self, caminho_corpus_base, remove_stopwords=True, normalization_method='lemma'):
        """
        Lê o corpus, processa os ficheiros TXT e guarda-os limpos em disco.
        """
        try:
            with open(caminho_corpus_base, 'r', encoding='utf-8') as f:
                corpus = json.load(f)
        except FileNotFoundError:
            print(f"[Erro] Ficheiro {caminho_corpus_base} não encontrado.")
            return

        print(f"\n[NLP] A normalizar textos e a criar ficheiros processados...")

        for doc_id, info in corpus.items():
            # Verificar se temos um ficheiro TXT para processar
            if info.get('has_pdf_txt') and info.get('pdf_txt_path'):
                # Caminho do TXT bruto (gerado pelo PDFExtractor)
                caminho_bruto = os.path.join(os.path.dirname(__file__), "..", "..", info['pdf_txt_path'])
                
                try:
                    with open(caminho_bruto, 'r', encoding='utf-8') as f:
                        texto_original = f.read()

                    # Limpeza e NLP (obtemos os tokens)
                    lang = 'portuguese' if 'por' in info.get('idioma', '') else 'english'
                    texto_limpo = self._limpar_texto_bruto(texto_original)
                    
                    tokens = self.nlp_processor.process_text(
                            texto_limpo, 
                            language=lang,
                            remove_stopwords=remove_stopwords,
                            normalization_method=normalization_method
                        )

                    safe_id = str(doc_id).replace("/", "_").replace("\\", "_")
                    caminho_tokens = os.path.join(self.processed_dir, f"{safe_id}_tokens.txt")

                    with open(caminho_tokens, 'w', encoding='utf-8') as f_out:
                        f_out.write(" ".join(tokens))
                    
                    info["ficheiro_tokens"] = caminho_tokens
                except Exception as e:
                    print(f"   [-] Erro no doc {doc_id}: {e}")
            
        return corpus