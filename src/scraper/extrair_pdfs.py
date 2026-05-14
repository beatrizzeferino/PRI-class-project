import json
import os
import requests
import subprocess
import tempfile
import re

class PDFExtractor:
    def __init__(self, output_dir="textos_pdfs"):
        """
        Inicializa o extrator de PDFs e garante que a pasta de destino existe.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _pdf_to_text(self, pdf_bytes, caminho_txt_final):
        """
        Converte os bytes do PDF baixado num ficheiro TXT físico.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_bytes)
            pdf_path = tmp_file.name
            
        subprocess.run(["pdftotext", pdf_path, caminho_txt_final], check=True)
        os.remove(pdf_path)

    def extrair_pdfs(self, corpus_file, output_file, limite=20):
        """
        Método principal: Lê o JSON (Corpus), processa os PDFs necessários e atualiza o JSON.
        """
        try:
            with open(corpus_file, "r", encoding="utf-8") as f:
                corpus = json.load(f) 
        except FileNotFoundError:
            print(f"[Erro] O ficheiro {corpus_file} não foi encontrado.")
            return

        n_pdfs_extraidos = sum(1 for doc in corpus.values() if doc.get("has_pdf_txt") is True)

        if n_pdfs_extraidos >= limite:
            print(f"Já foram previamente extraídos {limite} pdfs.")
            return

        count = n_pdfs_extraidos
        print(f"A iniciar... Faltam extrair {limite - count} PDFs.")

        for doc_id, doc in corpus.items():
            if count >= limite:
                break # Para o ciclo quando atingir o limite

            url = doc.get("link") or doc.get("url")

            # Ignorar se não tem link ou já tem o PDF extraído
            if not url or url == "N/A" or doc.get("has_pdf_txt"):
                continue

            try:
                print(f"[{count+1}/{limite}] A descarregar PDF para: {doc_id}...")

                # Fazer o download (com o disfarce de navegador)
                headers = {'User-Agent': 'Mozilla/5.0'}
                r = requests.get(url, headers=headers, timeout=15) 
                r.raise_for_status()

                #verifica se o ficheiro é um pdf
                if not r.content.startswith(b'%PDF'):
                    print("   -> [Aviso] O ficheiro descarregado não é um PDF válido. A ignorar...")
                    doc["has_pdf_txt"] = False
                    continue # Volta ao topo do ciclo e passa para o próximo artigo

                safe_doc_id = str(doc_id).replace("/", "_").replace("\\", "_")
                nome_ficheiro = f"{safe_doc_id}.txt"
                caminho_completo = os.path.join(self.output_dir, nome_ficheiro)

                if os.path.exists(caminho_completo):
                    if not doc.get("has_pdf_txt"):
                        doc["has_pdf_txt"] = True
                        doc["pdf_txt_path"] = caminho_completo
                    continue

                # Processar
                self._pdf_to_text(r.content, caminho_completo)

                # Atualizar Dicionário
                doc["has_pdf_txt"] = True
                doc["pdf_txt_path"] = caminho_completo

                print(f"   -> Guardado em: {caminho_completo}")
                count += 1

            except Exception as e:
                print(f"   -> [Erro] Falha ao processar: {e}")
                doc["has_pdf_txt"] = False 
            

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False, indent=4)
            
        print(f"\nProcesso concluído! Total de PDFs processados: {count}")


    