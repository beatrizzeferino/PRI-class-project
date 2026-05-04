import json
from src.search.nlp import TextProcessor

class CorpusProcessor:
    def __init__(self):
        self.nlp_processor = TextProcessor()
        self.documentos_processados = {}

    def processar_dataset(self, caminho_json, remove_stopwords=True, normalization_method='lemma'):
        try:
            with open(caminho_json, 'r', encoding='utf-8') as ficheiro:
                documentos_brutos = json.load(ficheiro)
        except FileNotFoundError:
            print(f"[Erro] Ficheiro {caminho_json} não encontrado.")
            return {}

        print(f"[Processador de corpus] A processar {len(documentos_brutos)} documentos...")

        for doc in documentos_brutos:
            doc_id = doc.get('doi')
            if not doc_id:
                continue

            iso_lang = doc.get('language', 'en').lower()
            lang_para_nlp = 'portuguese' if 'por' in iso_lang else 'english'

            texto_base = f"{doc.get('title', '')} {doc.get('abstract', '')}"
            if doc.get('keywords'):
                texto_base += " " + " ".join(doc['keywords'])

            if doc.get('pdf_txt'):
                texto_base += " " + doc['pdf_txt'] 

            tokens_limpos = self.nlp_processor.process_text(
                texto_base,
                language=lang_para_nlp,
                remove_stopwords=remove_stopwords,
                normalization_method=normalization_method
            )

            #PODEMOS ADICIONAR MAIS ITENS DEPENDENDO DO QUE VAI SER NECESSARIO
            self.documentos_processados[doc_id] = {
                "tokens_pesquisa": tokens_limpos,   # Para o TF-IDF
                "idioma": lang_para_nlp,
                "autores": doc.get('authors', []),  # Para Filtro por Autor
                "ano": doc.get('year', ''),         # Para Filtro por Ano
                "titulo": doc.get('title', ''),     # Para mostrar o resultado final
                "url": doc.get('url', '')           # Para o link final
            }

        print(f"[Indexer] Concluído! {len(self.documentos_processados)} documentos indexados.")
        return self.documentos_processados