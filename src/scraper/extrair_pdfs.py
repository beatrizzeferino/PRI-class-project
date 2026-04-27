import json
import os
import requests
import subprocess #para executar comandos como se fosse o terminal
import tempfile

def pdf_to_text(pdf):
    with tempfile.NamedTemporaryFile(delete= False, suffix=".pdf") as tmp_file: #cria um ficheiro temporario com a extensao .pdf, onde vamos guardar o pdf
        tmp_file.write(pdf)
        pdf_path =tmp_file.name #guarda o caminho do ficheiro temporario
        
    txt_path =pdf_path.replace(".pdf", ".txt")

    #executar pdftotext
    subprocess.run(["pdftotext", pdf_path, txt_path], check=True) #check permite que se falhar, lança erro automaticamente

    with open(txt_path, "r", encoding="utf-8", errors="ignorw") as f: #se houver erros, tipo caracteres estranhos ignora-os
        texto = f.read()

    #apagar ficheiros temporarios
    os.remove(pdf_path)
    os.remove(txt_path)

    return texto


def add_texto_scraper(scraper_file, output_file, limite):
    with open(scraper_file, "r", encoding="utf-8") as f:
        scraper= json.load(f)
    
    n_pdfs_extraidos = sum(1 for doc in scraper if doc.get("pdf_txt") is not None)

    if n_pdfs_extraidos >= limite:
        print("Já foram previamente extraidos 20 pdfs.")
        return

    count = n_pdfs_extraidos
    i=0
    while i< len(scraper) and count <limite:
        doc= scraper[i]
        i+=1
        url = doc.get("document_link")

        #se nao captou o link para o pdf
        if url=="N/A":
            continue

        try:
            print(f"A transformar PDF: {doc.get('title')}")

            r = requests.get(url)
            r.raise_for_status()

            texto = pdf_to_text(r.content)
            doc["pdf_txt"] = texto

            count += 1

        except Exception as e:
            print(f"Erro no documento {doc.get('title')}: {e}")
            doc["pdf_txt"] = None

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(scraper, f, ensure_ascii=False, indent=2)