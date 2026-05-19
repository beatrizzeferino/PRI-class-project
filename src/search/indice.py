import json
import math
import os
import re
from collections import defaultdict


# ══════════════════════════════════════════════════════════════════════════════
#  PostingList
# ══════════════════════════════════════════════════════════════════════════════

class PostingList:
    """Lista de postings para um único termo do vocabulário."""

    def __init__(self):
        self.postings = []
        self.skip_pointers = {}
        self.df = 0

    def adicionar_posting(self, doc_id: str, tf: int):

        self.postings.append({
            "doc_id": doc_id,
            "tf": tf
        })

        self.df += 1

    def ordenar(self):
        self.postings.sort(key=lambda x: x["doc_id"])

    def construir_skip_pointers(self):
        self.skip_pointers = {}

        n = len(self.postings)

        if n <= 3:
            return

        salto = int(math.sqrt(n))

        for i in range(0, n - salto, salto):
            self.skip_pointers[i] = i + salto

    def doc_ids(self):
        return [p["doc_id"] for p in self.postings]

    def __len__(self):
        return len(self.postings)


# ══════════════════════════════════════════════════════════════════════════════
#  IndiceInvertido
# ══════════════════════════════════════════════════════════════════════════════

class IndiceInvertido:

    def __init__(self):

        self.indice = {}
        self.documentos = {}
        self.num_documentos=0

    # ──────────────────────────────────────────────────────────────────
    #  Tokenização
    # ──────────────────────────────────────────────────────────────────

    def _tokenizar(self, texto: str):
        return re.findall(r'\b\w+\b', texto.lower())

    # ──────────────────────────────────────────────────────────────────
    #  Reset
    # ──────────────────────────────────────────────────────────────────

    def reset_total(self):

        self.indice = {}
        self.documentos = {}
        self.num_documentos = 0
        self._normas = {}
        self._normas_validas = False

        print("[Reset] Todos os dados foram limpos da memória.")

    # ──────────────────────────────────────────────────────────────────
    #  Construção
    # ──────────────────────────────────────────────────────────────────

    def construir(self, documentos_processados, pasta_textos=None):

        for doc_id, info in documentos_processados.items():

            if pasta_textos:

                tokens_pdf = self._carregar_tokens_pdf(doc_id, pasta_textos)

                if tokens_pdf:
                    info = dict(info)

                    info["tokens_pesquisa"] = (
                        info.get("tokens_pesquisa", []) + tokens_pdf
                    )

            self._indexar_documento(doc_id, info)

        self._finalizar_indice()

        print(
            f"[Índice] Construído com {self.num_documentos} documentos "
            f"e {len(self.indice)} termos únicos."
        )

    # ──────────────────────────────────────────────────────────────────
    #  PDF/TXT
    # ──────────────────────────────────────────────────────────────────

    def _carregar_tokens_pdf(self, doc_id, pasta_textos):

        nome = doc_id.replace("/", "_") + "_tokens.txt"

        caminho = os.path.join(pasta_textos, nome)

        if not os.path.isfile(caminho):
            print(f"  [txt] NAO encontrado: {caminho}")
            return []

        try:

            with open(caminho, "r", encoding="utf-8", errors="ignore") as f:
                texto = f.read()

            tokens = self._tokenizar(texto)
            print(f"  [txt] Lido: {caminho}  ({len(tokens)} tokens)")

            return tokens

        except OSError:
            return []

    # ──────────────────────────────────────────────────────────────────
    #  Indexação
    # ──────────────────────────────────────────────────────────────────

    def _indexar_documento(self, doc_id, info):

        self.documentos[doc_id] = {
            "titulo": info.get("titulo", ""),
            "url": info.get("url", ""),
            "autores": info.get("autores", []),
            "ano": info.get("ano", ""),
            "idioma": info.get("idioma", "english"),
            "abstrato": info.get("abstrato", ""),
            "link": info.get("link", "N/A"),
        }

        tokens = info.get("tokens_pesquisa", [])

        tf_doc = defaultdict(int)

        for token in tokens:
            tf_doc[token.lower()] += 1

        for termo, tf in tf_doc.items():

            if termo not in self.indice:
                self.indice[termo] = PostingList()

            self.indice[termo].adicionar_posting(doc_id, tf)

        self.num_documentos = len(self.documentos)

        self._normas_validas = False

    def _finalizar_indice(self):

        for pl in self.indice.values():
            pl.ordenar()
            pl.construir_skip_pointers()

    # ──────────────────────────────────────────────────────────────────
    #  Atualização incremental
    # ──────────────────────────────────────────────────────────────────

    def adicionar_documentos(self, novos_docs, pasta_textos=None):

        termos_afetados = set()
        adicionados = 0

        for doc_id, info in novos_docs.items():

            if doc_id in self.documentos:
                print(f"  [Aviso] '{doc_id}' já existe. A ignorar.")
                continue

            if pasta_textos:

                tokens_pdf = self._carregar_tokens_pdf(doc_id, pasta_textos)

                if tokens_pdf:
                    info = dict(info)

                    info["tokens_pesquisa"] = (
                        info.get("tokens_pesquisa", []) + tokens_pdf
                    )

            self._indexar_documento(doc_id, info)

            termos_afetados.update(
                t.lower() for t in info.get("tokens_pesquisa", [])
            )

            adicionados += 1

        for termo in termos_afetados:

            if termo in self.indice:
                self.indice[termo].ordenar()
                self.indice[termo].construir_skip_pointers()

        self._normas_validas = False

        print(
            f"[Índice] {adicionados} documento(s) adicionado(s). "
            f"Total: {self.num_documentos}"
        )

    # ──────────────────────────────────────────────────────────────────
    #  Posting Lists
    # ──────────────────────────────────────────────────────────────────

    def obter_posting_list(self, termo):
        return self.indice.get(termo.lower())

    def intersetar_com_skip(self, lista1, lista2):

        resultado = PostingList()

        i = 0
        j = 0

        p1 = lista1.postings
        p2 = lista2.postings

        while i < len(p1) and j < len(p2):

            if p1[i]["doc_id"] == p2[j]["doc_id"]:

                resultado.adicionar_posting(
                    p1[i]["doc_id"],
                    p1[i]["tf"] + p2[j]["tf"]
                )

                i += 1
                j += 1

            elif p1[i]["doc_id"] < p2[j]["doc_id"]:

                if (
                    i in lista1.skip_pointers and
                    p1[lista1.skip_pointers[i]]["doc_id"] <= p2[j]["doc_id"]
                ):
                    i = lista1.skip_pointers[i]
                else:
                    i += 1

            else:

                if (
                    j in lista2.skip_pointers and
                    p2[lista2.skip_pointers[j]]["doc_id"] <= p1[i]["doc_id"]
                ):
                    j = lista2.skip_pointers[j]
                else:
                    j += 1

        return resultado

        # ──────────────────────────────────────────────────────────────────
    #  Estatísticas
    # ──────────────────────────────────────────────────────────────────

    def estatisticas(self) -> dict:
        """Estatísticas gerais do índice."""
        return {
            "num_documentos":       self.num_documentos,
            "num_termos_unicos":    len(self.indice),
            "top_10_termos_por_df": sorted(
                [(t, pl.df) for t, pl in self.indice.items()],
                key=lambda x: x[1], reverse=True
            )[:10],
        }
    
    # ──────────────────────────────────────────────────────────────────
    #  Persistência
    # ──────────────────────────────────────────────────────────────────

    def guardar(self, caminho: str):
        """
        Serializa o índice para JSON — apenas num_documentos e indice.
        Os metadados (self.documentos) e os skip pointers NÃO são guardados.
        Os skip pointers são reconstruídos automaticamente no carregar().
        """
        dados = {
            "num_documentos": self.num_documentos,
            "indice": {
                termo: {"df": pl.df, "postings": pl.postings}
                for termo, pl in self.indice.items()
            },
        }
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)
        print(f"[OK] Índice guardado em: {caminho}  "
              f"({self.num_documentos} docs, {len(self.indice)} termos)")

    def carregar(self, caminho: str):
        """
        Carrega índice de um ficheiro JSON guardado com guardar().
        Reconstrói os skip pointers automaticamente.
        Nota: self.documentos (metadados) NÃO é persistido — fica vazio após
        carregar. Funções que dependem de metadados (pesquisa_por_autor,
        pesquisa_tfidf, negar) só funcionam se o índice foi construído
        nesta sessão ou se os metadados forem re-injectados manualmente.
        """
        with open(caminho, "r", encoding="utf-8") as f:
            dados = json.load(f)

        self.num_documentos = dados["num_documentos"]
        self.documentos = {}   # metadados não são persistidos
        self.indice = {}

        for termo, info in dados["indice"].items():
            pl = PostingList()
            pl.postings = info["postings"]
            pl.df = info["df"]
            pl.construir_skip_pointers()
            self.indice[termo] = pl

        self._normas_validas = False
        print(f"[OK] Índice carregado: {self.num_documentos} documentos, "
              f"{len(self.indice)} termos.")

