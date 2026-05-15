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
        for p in self.postings:
            if p["doc_id"] == doc_id:
                p["tf"] = tf
                return

        self.postings.append({"doc_id": doc_id, "tf": tf})
        self.df = len(self.postings)

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
        self.num_documentos = 0

        self._normas = {}
        self._normas_validas = False

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

    def unir(self, lista1, lista2):

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

                resultado.adicionar_posting(
                    p1[i]["doc_id"],
                    p1[i]["tf"]
                )

                i += 1

            else:

                resultado.adicionar_posting(
                    p2[j]["doc_id"],
                    p2[j]["tf"]
                )

                j += 1

        while i < len(p1):
            resultado.adicionar_posting(p1[i]["doc_id"], p1[i]["tf"])
            i += 1

        while j < len(p2):
            resultado.adicionar_posting(p2[j]["doc_id"], p2[j]["tf"])
            j += 1

        return resultado

    def negar(self, lista):

        ids_existentes = set(lista.doc_ids())

        resultado = PostingList()

        for doc_id in sorted(self.documentos.keys()):

            if doc_id not in ids_existentes:
                resultado.adicionar_posting(doc_id, 0)

        resultado.construir_skip_pointers()

        return resultado

    # ──────────────────────────────────────────────────────────────────
    #  Pesquisa booleana
    # ──────────────────────────────────────────────────────────────────

    def pesquisa_booleana(self, query):

        tokens = self._tokenizar_query(query)

        pl = self._avaliar_expressao(tokens)

        return pl.doc_ids() if pl else []

    def _tokenizar_query(self, query):

        partes = re.findall(
            r'\(|\)|AND|OR|NOT|[^\s()]+',
            query,
            re.IGNORECASE
        )

        resultado = []

        for parte in partes:

            upper = parte.upper()

            if upper in ("AND", "OR", "NOT", "(", ")"):
                token_atual = upper
            else:
                token_atual = parte.lower()

            if resultado:

                anterior = resultado[-1]

                precisa_and = (
                    anterior not in ("AND", "OR", "NOT", "(")
                    and token_atual not in ("AND", "OR", ")")
                )

                if precisa_and:
                    resultado.append("AND")

            resultado.append(token_atual)

        return resultado

    def _avaliar_expressao(self, tokens):

        pos = [0]

        def avaliar_or():

            esq = avaliar_and()

            while pos[0] < len(tokens) and tokens[pos[0]] == "OR":

                pos[0] += 1

                dir_ = avaliar_and()

                esq = self.unir(esq, dir_)

            return esq

        def avaliar_and():

            esq = avaliar_not()

            while pos[0] < len(tokens) and tokens[pos[0]] == "AND":

                pos[0] += 1

                dir_ = avaliar_not()

                esq = self.intersetar_com_skip(esq, dir_)

            return esq

        def avaliar_not():

            if pos[0] < len(tokens) and tokens[pos[0]] == "NOT":

                pos[0] += 1

                operando = avaliar_primario()

                return self.negar(operando)

            return avaliar_primario()

        def avaliar_primario():

            if pos[0] >= len(tokens):
                return PostingList()

            tok = tokens[pos[0]]

            if tok == "(":

                pos[0] += 1

                resultado = avaliar_or()

                if pos[0] < len(tokens) and tokens[pos[0]] == ")":
                    pos[0] += 1

                return resultado

            else:

                pos[0] += 1

                pl = self.obter_posting_list(tok)

                return pl if pl else PostingList()

        return avaliar_or()

    # ──────────────────────────────────────────────────────────────────
    #  TF-IDF
    # ──────────────────────────────────────────────────────────────────

    def calcular_tfidf(self, doc_id, termo):

        pl = self.obter_posting_list(termo)

        if pl is None or pl.df == 0:
            return 0.0

        tf_bruto = next(
            (
                p["tf"]
                for p in pl.postings
                if p["doc_id"] == doc_id
            ),
            0
        )

        if tf_bruto == 0:
            return 0.0

        tf = 1 + math.log(tf_bruto)

        idf = math.log((1 + self.num_documentos) / (1 + pl.df)) + 1

        return tf * idf

    def _calcular_normas(self):

        normas = defaultdict(float)

        N = self.num_documentos

        if N == 0:
            self._normas = {}
            self._normas_validas = True
            return

        for termo, pl in self.indice.items():

            idf = (
                math.log((1 + N) / (1 + pl.df)) + 1
                if pl.df > 0 else 0.0
            )

            for p in pl.postings:

                tf = 1 + math.log(p["tf"])

                normas[p["doc_id"]] += (tf * idf) ** 2

        self._normas = {
            doc_id: math.sqrt(v)
            for doc_id, v in normas.items()
        }

        self._normas_validas = True

    def pesquisa_tfidf(self, query, top_k=10):

        if not self._normas_validas:
            self._calcular_normas()

        termos_query = self._tokenizar(query)

        if not termos_query:
            return []

        scores = defaultdict(float)

        N = self.num_documentos

        tf_query = defaultdict(int)

        for t in termos_query:
            tf_query[t] += 1

        for termo, tf_q in tf_query.items():

            pl = self.obter_posting_list(termo)

            if pl is None or pl.df == 0:
                continue

            idf = math.log((1 + N) / (1 + pl.df)) + 1

            tfidf_q = (1 + math.log(tf_q)) * idf

            for posting in pl.postings:

                tf_d = 1 + math.log(posting["tf"])

                tfidf_d = tf_d * idf

                scores[posting["doc_id"]] += tfidf_d * tfidf_q

        resultados = []

        for doc_id, score in scores.items():

            norma = self._normas.get(doc_id, 1.0)

            score_norm = score / norma if norma > 0 else 0.0

            meta = self.documentos.get(doc_id, {})

            resultados.append({
                "doc_id": doc_id,
                "score": round(score_norm, 6),
                "titulo": meta.get("titulo", ""),
                "autores": meta.get("autores", []),
                "ano": meta.get("ano", ""),
                "url": meta.get("url", ""),
                "link": meta.get("link", "N/A")
            })

        resultados.sort(key=lambda x: x["score"], reverse=True)

        return resultados[:top_k]
    
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

