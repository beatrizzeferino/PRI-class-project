import json
from src.search.nlp import TextProcessor

class CorpusProcessor:
    def __init__(self):
        self.nlp_processor = TextProcessor()
        self.documentos_processados = {}

    def guardar_json(self, caminho_saida):
        """
        Guarda o dicionário de documentos processados num ficheiro JSON.
        """
        try:
            with open(caminho_saida, 'w', encoding='utf-8') as f:
                json.dump(self.documentos_processados, f, ensure_ascii=False, indent=4)
            print(f"[Corpus] Guardado com sucesso em: {caminho_saida}")
        except Exception as e:
            print(f"[Erro] Falha ao guardar JSON: {e}")

    def processar_dataset(self, caminho_json, remove_stopwords=True, normalization_method='lemma',caminho_saida='processed_corpus.json'):
        try:
            with open(caminho_json, 'r', encoding='utf-8') as ficheiro:
                documentos_brutos = json.load(ficheiro)
        except FileNotFoundError:
            print(f"[Erro] Ficheiro {caminho_json} não encontrado.")
            return {}

        print(f"[Processador de corpus] A processar {len(documentos_brutos)} documentos...")

        doc_counter=0
        for doc in documentos_brutos:
            base_id = doc.get('doi')

            if doc.get('doi')=="N/A": #tratar dos casos onde o documento não tem doi associado
                base_id = f"doc_{doc_counter}"
                doc_counter += 1

            doc_id = base_id

            if doc_id in self.documentos_processados: #para o caso de haver dois ou mais documentos com o mesmo doi, o identificador fica por exempo doi_1, doi_2,...
                doc_id = f"{base_id}_{doc_counter}"
                doc_counter += 1

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

            self.documentos_processados[doc_id] = {
                "tokens_pesquisa": tokens_limpos, 
                "titulo": doc.get('title',''),
                "ano": doc.get('year', ''),
                "doi": doc.get('doi',''),
                "abstrato": doc.get('abstract',''),
                "autores": doc.get('authors', []), 
                "url": doc.get('url', ''), 
                "keywords": doc.get('keywords',[]),
                "relations": doc.get('relations',[]),
                "idioma": lang_para_nlp,
                "link": doc.get('document_link','')   
                          
            }

        print(f"[Indexer] Concluído! {len(self.documentos_processados)} documentos indexados.")
        self.guardar_json(caminho_saida)
        return self.documentos_processados
