"""
indice.py — Índice Invertido para Motor de Pesquisa de Publicações Científicas
Universidade do Minho — Pesquisa e Recuperação de Informação 2025/2026

Compatível com:
  - scraper_results.json  (campos: title, year, doi, abstract, authors, url, document_link)
  - processed_corpus.json (campos: titulo, ano, doi, abstrato, autores, url, tokens_pesquisa, idioma)
  - textos_processados/   (ficheiros .txt cujo nome é o DOI com '/' substituído por '_')
"""

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
        self.postings = []       # [{"doc_id": str, "tf": int}, ...]
        self.skip_pointers = {}  # {índice_i: índice_j}  (saltos de sqrt(n))
        self.df = 0              # nº de documentos que contêm o termo

    # ------------------------------------------------------------------
    def adicionar_posting(self, doc_id: str, tf: int):
        """Adiciona ou actualiza o posting de um documento."""
        for p in self.postings:
            if p["doc_id"] == doc_id:
                p["tf"] = tf
                return
        self.postings.append({"doc_id": doc_id, "tf": tf})
        self.df = len(self.postings)

    def ordenar(self):
        """Ordena por doc_id (necessário para merge e skip pointers)."""
        self.postings.sort(key=lambda x: x["doc_id"])

    def construir_skip_pointers(self):
        """
        Constrói skip pointers com salto de sqrt(n).
        Só faz sentido para listas com mais de 3 entradas.
        """
        self.skip_pointers = {}
        n = len(self.postings)
        if n <= 3:
            return
        salto = int(math.sqrt(n))
        for i in range(0, n - salto, salto):
            self.skip_pointers[i] = i + salto

    def doc_ids(self) -> list:
        """Lista de doc_ids (útil para o modelo booleano e debug)."""
        return [p["doc_id"] for p in self.postings]

    def __len__(self):
        return len(self.postings)

    def __repr__(self):
        return f"PostingList(df={self.df}, docs={self.doc_ids()[:5]}{'...' if self.df > 5 else ''})"


# ══════════════════════════════════════════════════════════════════════════════
#  IndiceInvertido
# ══════════════════════════════════════════════════════════════════════════════

class IndiceInvertido:
    """
    Índice invertido com suporte a:
      • Construção a partir de scraper_results.json ou processed_corpus.json
      • Indexação opcional do texto completo dos PDFs (pasta textos_processados/)
      • Pesquisa booleana: AND, OR, NOT (com precedência correta)
      • Cálculo de TF-IDF e ranking por similaridade do cosseno
      • Actualização incremental
      • Persistência JSON (guardar / carregar)
    """

    def __init__(self):
        self.indice: dict[str, PostingList] = {}
        self.documentos: dict[str, dict] = {}   # doc_id → metadados
        self.num_documentos: int = 0
        # Comprimento do vector TF-IDF de cada documento (para cosseno)
        self._normas: dict[str, float] = {}
        self._normas_validas = False             # invalida ao adicionar docs

    # ──────────────────────────────────────────────────────────────────
    #  Reset
    # ──────────────────────────────────────────────────────────────────

    def reset_total(self):
        """Limpa tudo da memória."""
        self.indice = {}
        self.documentos = {}
        self.num_documentos = 0
        self._normas = {}
        self._normas_validas = False
        print("[Reset] Todos os dados foram limpos da memória.")

    # ──────────────────────────────────────────────────────────────────
    #  Construção — entrada principal
    # ──────────────────────────────────────────────────────────────────

    def construir_de_scraper(self,
                             scraper_results: list,
                             pasta_textos: str | None = None):
        """
        Constrói o índice a partir do scraper_results.json.

        Parâmetros:
            scraper_results  : lista de dicts com campos title/year/doi/abstract/authors/url/document_link
            pasta_textos     : caminho para a pasta textos_processados/ (opcional).
                               Quando fornecido, o texto completo dos PDFs é também indexado.
        """
        for doc in scraper_results:
            doc_id = doc.get("doi") or doc.get("url") or doc.get("title", "sem_id")
            # Normalizar para o formato interno
            info = self._normalizar_scraper(doc, pasta_textos)
            self._indexar_documento(doc_id, info)

        self._finalizar_indice()
        print(f"[Índice] Construído com {self.num_documentos} documentos "
              f"e {len(self.indice)} termos únicos.")

    def construir_de_processed_corpus(self,
                                      documentos_processados: dict,
                                      pasta_textos: str | None = None):
        """
        Constrói o índice a partir do processed_corpus.json
        (formato: {doi: {titulo, autores, tokens_pesquisa, ...}}).

        Parâmetros:
            documentos_processados : dict devolvido pelo CorpusProcesser
            pasta_textos           : caminho para textos_processados/ (opcional)
        """
        for doc_id, info in documentos_processados.items():
            # Se houver texto do PDF disponível, acrescenta os seus tokens
            if pasta_textos:
                tokens_pdf = self._carregar_tokens_pdf(doc_id, pasta_textos)
                if tokens_pdf:
                    info = dict(info)           # cópia para não alterar original
                    info["tokens_pesquisa"] = info.get("tokens_pesquisa", []) + tokens_pdf
            self._indexar_documento(doc_id, info)

        self._finalizar_indice()
        print(f"[Índice] Construído com {self.num_documentos} documentos "
              f"e {len(self.indice)} termos únicos.")

    # Mantém compatibilidade com o nome antigo
    def construir_de_indexer(self, documentos_processados: dict,
                              pasta_textos: str | None = None):
        """Alias de construir_de_processed_corpus (compatibilidade)."""
        self.construir_de_processed_corpus(documentos_processados, pasta_textos)

    # ──────────────────────────────────────────────────────────────────
    #  Normalização scraper → formato interno
    # ──────────────────────────────────────────────────────────────────

    def _normalizar_scraper(self, doc: dict, pasta_textos: str | None) -> dict:
        """
        Converte um registo do scraper_results para o formato interno
        e tokeniza o título + abstract para indexação.
        """
        doi = doc.get("doi", "")
        titulo = doc.get("title", "")
        abstract = doc.get("abstract", "")
        ano = doc.get("year", "")
        # Normaliza ano: "2022-01-17" → "2022"
        if ano and "-" in str(ano):
            ano = str(ano).split("-")[0]

        # Tokenização simples: título + abstract
        texto_base = f"{titulo} {abstract}"
        tokens = self._tokenizar(texto_base)

        # Texto completo do PDF (se disponível)
        if pasta_textos and doi:
            tokens_pdf = self._carregar_tokens_pdf(doi, pasta_textos)
            tokens += tokens_pdf

        return {
            "titulo":          titulo,
            "url":             doc.get("url", ""),
            "autores":         doc.get("authors", []),
            "ano":             ano,
            "idioma":          "english",   # scraper não tem campo idioma
            "abstrato":        abstract,
            "link":            doc.get("document_link", "N/A"),
            "tokens_pesquisa": tokens,
        }

    # ──────────────────────────────────────────────────────────────────
    #  Tokenização
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenizar(texto: str) -> list[str]:
        """
        Tokenização básica: minúsculas, apenas palavras alfanuméricas,
        descarta tokens com < 2 caracteres.
        (Se o CorpusProcesser já fez stemming/lematização, usa os seus tokens.)
        """
        if not texto:
            return []
        tokens = re.findall(r"[a-zA-ZÀ-ÿ0-9]+", texto.lower())
        return [t for t in tokens if len(t) >= 2]

    # ──────────────────────────────────────────────────────────────────
    #  Leitura de texto de PDF
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _doi_para_nome_ficheiro(doi: str) -> str:
        """Converte DOI para nome de ficheiro: '/' → '_'  (convenção do projeto)."""
        return doi.replace("/", "_")

    def _carregar_tokens_pdf(self, doc_id: str, pasta_textos: str) -> list[str]:
        """
        Carrega e tokeniza o ficheiro .txt correspondente ao doc_id (DOI).
        Retorna lista de tokens ou [] se o ficheiro não existir.
        """
        nome = self._doi_para_nome_ficheiro(doc_id) + "_tokens.txt"
        caminho = os.path.join(pasta_textos, nome)
        if not os.path.isfile(caminho):
            print(f"  [txt] NAO encontrado: {caminho}")
            return []
        try:
            with open(caminho, "r", encoding="utf-8", errors="ignore") as f:
                texto = f.read()
            tokens = IndiceInvertido._tokenizar(texto)
            print(f"  [txt] Lido: {caminho}  ({len(tokens)} tokens)")
            return tokens
        except OSError:
            return []

    # ──────────────────────────────────────────────────────────────────
    #  Indexação interna
    # ──────────────────────────────────────────────────────────────────

    def _indexar_documento(self, doc_id: str, info: dict):
        """
        Indexa um único documento:
          1. Guarda metadados
          2. Calcula TF bruto por token
          3. Actualiza as PostingLists
        """
        self.documentos[doc_id] = {
            "titulo":   info.get("titulo", ""),
            "url":      info.get("url", ""),
            "autores":  info.get("autores", []),
            "ano":      info.get("ano", ""),
            "idioma":   info.get("idioma", "english"),
            "abstrato": info.get("abstrato", ""),
            "link":     info.get("link", "N/A"),
        }

        tokens = info.get("tokens_pesquisa", [])
        tf_doc: dict[str, int] = defaultdict(int)
        for token in tokens:
            tf_doc[token.lower()] += 1

        for termo, tf in tf_doc.items():
            if termo not in self.indice:
                self.indice[termo] = PostingList()
            self.indice[termo].adicionar_posting(doc_id, tf)

        self.num_documentos = len(self.documentos)
        self._normas_validas = False   # normas TF-IDF precisam de ser recalculadas

    def _finalizar_indice(self):
        """Ordena todas as listas e reconstrói skip pointers."""
        for pl in self.indice.values():
            pl.ordenar()
            pl.construir_skip_pointers()

    # ──────────────────────────────────────────────────────────────────
    #  Actualização incremental
    # ──────────────────────────────────────────────────────────────────

    def adicionar_documentos(self, novos_docs, pasta_textos: str | None = None,
                              formato: str = "processed_corpus"):
        """
        Adiciona documentos a um índice existente sem reconstruir do zero.

        Parâmetros:
            novos_docs    : lista (scraper) ou dict (processed_corpus)
            pasta_textos  : pasta com ficheiros .txt (opcional)
            formato       : "scraper" | "processed_corpus"
        """
        termos_afetados: set[str] = set()

        itens = novos_docs.items() if isinstance(novos_docs, dict) else [
            (d.get("doi") or d.get("url") or d.get("title", "sem_id"), d)
            for d in novos_docs
        ]

        adicionados = 0
        for doc_id, raw in itens:
            if doc_id in self.documentos:
                print(f"  [Aviso] '{doc_id}' já existe. A ignorar.")
                continue

            if formato == "scraper":
                info = self._normalizar_scraper(raw, pasta_textos)
            else:
                info = dict(raw)
                if pasta_textos:
                    info["tokens_pesquisa"] = info.get("tokens_pesquisa", []) + \
                                              self._carregar_tokens_pdf(doc_id, pasta_textos)

            self._indexar_documento(doc_id, info)
            termos_afetados.update(t.lower() for t in info.get("tokens_pesquisa", []))
            adicionados += 1

        for termo in termos_afetados:
            if termo in self.indice:
                self.indice[termo].ordenar()
                self.indice[termo].construir_skip_pointers()

        print(f"[Índice] {adicionados} documento(s) adicionado(s). Total: {self.num_documentos}")

    # ──────────────────────────────────────────────────────────────────
    #  Operações sobre PostingLists
    # ──────────────────────────────────────────────────────────────────

    def obter_posting_list(self, termo: str) -> PostingList | None:
        """Devolve a PostingList de um termo (None se não existir)."""
        return self.indice.get(termo.lower())

    def intersetar_com_skip(self, lista1: PostingList,
                             lista2: PostingList) -> PostingList:
        """AND com skip pointers. TF combinado = soma dos dois TFs."""
        resultado = PostingList()
        i = j = 0
        p1, p2 = lista1.postings, lista2.postings

        while i < len(p1) and j < len(p2):
            if p1[i]["doc_id"] == p2[j]["doc_id"]:
                resultado.adicionar_posting(p1[i]["doc_id"],
                                            p1[i]["tf"] + p2[j]["tf"])
                i += 1
                j += 1
            elif p1[i]["doc_id"] < p2[j]["doc_id"]:
                if (i in lista1.skip_pointers and
                        p1[lista1.skip_pointers[i]]["doc_id"] <= p2[j]["doc_id"]):
                    i = lista1.skip_pointers[i]
                else:
                    i += 1
            else:
                if (j in lista2.skip_pointers and
                        p2[lista2.skip_pointers[j]]["doc_id"] <= p1[i]["doc_id"]):
                    j = lista2.skip_pointers[j]
                else:
                    j += 1

        return resultado

    def unir(self, lista1: PostingList, lista2: PostingList) -> PostingList:
        """OR — merge linear, TF combinado = soma."""
        resultado = PostingList()
        i = j = 0
        p1, p2 = lista1.postings, lista2.postings

        while i < len(p1) and j < len(p2):
            if p1[i]["doc_id"] == p2[j]["doc_id"]:
                resultado.adicionar_posting(p1[i]["doc_id"],
                                            p1[i]["tf"] + p2[j]["tf"])
                i += 1
                j += 1
            elif p1[i]["doc_id"] < p2[j]["doc_id"]:
                resultado.adicionar_posting(p1[i]["doc_id"], p1[i]["tf"])
                i += 1
            else:
                resultado.adicionar_posting(p2[j]["doc_id"], p2[j]["tf"])
                j += 1

        while i < len(p1):
            resultado.adicionar_posting(p1[i]["doc_id"], p1[i]["tf"])
            i += 1
        while j < len(p2):
            resultado.adicionar_posting(p2[j]["doc_id"], p2[j]["tf"])
            j += 1

        return resultado

    def negar(self, lista: PostingList) -> PostingList:
        """
        NOT — devolve todos os documentos que NÃO estão em lista.
        Requer que self.documentos esteja populado.
        """
        ids_existentes = set(lista.doc_ids())
        resultado = PostingList()
        for doc_id in sorted(self.documentos.keys()):
            if doc_id not in ids_existentes:
                resultado.adicionar_posting(doc_id, 0)
        resultado.construir_skip_pointers()
        return resultado

    # ──────────────────────────────────────────────────────────────────
    #  Pesquisa booleana de alto nível
    # ──────────────────────────────────────────────────────────────────

    def pesquisa_booleana(self, query: str) -> list[str]:
        """
        Avalia uma expressão booleana e devolve lista de doc_ids.

        Suporta:
          • AND  (explícito ou implícito por espaço)
          • OR
          • NOT
          • Parênteses para agrupamento

        Exemplos:
            "machine learning"          → AND implícito
            "neural AND network"
            "python OR java"
            "NOT python"
            "(neural OR deep) AND learning"
        """
        tokens = self._tokenizar_query(query)
        pl = self._avaliar_expressao(tokens)
        return pl.doc_ids() if pl else []

    def _tokenizar_query(self, query: str) -> list[str]:
        """
        Divide a query em tokens: palavras, AND, OR, NOT, (, ).
        Espaços entre palavras (sem operador) são convertidos em AND implícito.
        """
        # Separar por operadores e parênteses
        partes = re.findall(r'\(|\)|AND|OR|NOT|[^\s()]+', query, re.IGNORECASE)
        resultado = []
        for i, parte in enumerate(partes):
            upper = parte.upper()
            if upper in ("AND", "OR", "NOT", "(", ")"):
                resultado.append(upper)
            else:
                # Inserir AND implícito se necessário
                if resultado and resultado[-1] not in ("AND", "OR", "NOT", "("):
                    resultado.append("AND")
                resultado.append(parte.lower())
        return resultado

    def _avaliar_expressao(self, tokens: list[str]) -> PostingList:
        """
        Parser de expressão booleana por precedência:
          NOT > AND > OR
        Usa o algoritmo de precedência de operadores (shunting-yard simplificado).
        """
        # Converte para pilha e avalia recursivamente
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
    #  TF-IDF e similaridade do cosseno
    # ──────────────────────────────────────────────────────────────────

    def calcular_tfidf(self, doc_id: str, termo: str) -> float:
        """
        TF-IDF logarítmico:
          TF  = 1 + log(tf_bruto)   se tf_bruto > 0, senão 0
          IDF = log(N / df)
          score = TF * IDF
        """
        pl = self.obter_posting_list(termo)
        if pl is None or pl.df == 0:
            return 0.0

        tf_bruto = next((p["tf"] for p in pl.postings if p["doc_id"] == doc_id), 0)
        if tf_bruto == 0:
            return 0.0

        tf = 1 + math.log(tf_bruto)
        idf = math.log(self.num_documentos / pl.df)
        return tf * idf

    def _calcular_normas(self):
        """Pré-calcula o comprimento do vector TF-IDF de cada documento."""
        normas: dict[str, float] = defaultdict(float)
        N = self.num_documentos
        if N == 0:
            self._normas = {}
            self._normas_validas = True
            return

        for termo, pl in self.indice.items():
            idf = math.log(N / pl.df) if pl.df > 0 else 0.0
            for p in pl.postings:
                tf = 1 + math.log(p["tf"]) if p["tf"] > 0 else 0.0
                normas[p["doc_id"]] += (tf * idf) ** 2

        self._normas = {doc_id: math.sqrt(v) for doc_id, v in normas.items()}
        self._normas_validas = True

    def pesquisa_tfidf(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Pesquisa por texto livre com ranking TF-IDF + cosseno.

        Devolve lista de dicts ordenados por score descendente:
          [{"doc_id": ..., "score": ..., "titulo": ..., "autores": ..., "ano": ..., "url": ...}, ...]

        Parâmetros:
            query  : texto livre (os tokens são tratados como AND implícito para o score)
            top_k  : número máximo de resultados a devolver
        """
        if not self._normas_validas:
            self._calcular_normas()

        termos_query = self._tokenizar(query)
        if not termos_query:
            return []

        # Acumula scores: doc_id → soma(tfidf_doc * tfidf_query)
        scores: dict[str, float] = defaultdict(float)
        N = self.num_documentos

        # TF da query
        tf_query: dict[str, int] = defaultdict(int)
        for t in termos_query:
            tf_query[t] += 1

        for termo, tf_q in tf_query.items():
            pl = self.obter_posting_list(termo)
            if pl is None or pl.df == 0:
                continue
            idf = math.log(N / pl.df)
            tfidf_q = (1 + math.log(tf_q)) * idf

            for posting in pl.postings:
                tf_d = 1 + math.log(posting["tf"]) if posting["tf"] > 0 else 0.0
                tfidf_d = tf_d * idf
                scores[posting["doc_id"]] += tfidf_d * tfidf_q

        # Normalizar pelo comprimento do vector do documento (cosseno)
        resultados = []
        for doc_id, score in scores.items():
            norma = self._normas.get(doc_id, 1.0)
            score_norm = score / norma if norma > 0 else 0.0
            meta = self.documentos.get(doc_id, {})
            resultados.append({
                "doc_id":  doc_id,
                "score":   round(score_norm, 6),
                "titulo":  meta.get("titulo", ""),
                "autores": meta.get("autores", []),
                "ano":     meta.get("ano", ""),
                "url":     meta.get("url", ""),
                "link":    meta.get("link", "N/A"),
            })

        resultados.sort(key=lambda x: x["score"], reverse=True)
        return resultados[:top_k]

    # ──────────────────────────────────────────────────────────────────
    #  Pesquisa por autor
    # ──────────────────────────────────────────────────────────────────

    def pesquisa_por_autor(self, nome_autor: str) -> list[dict]:
        """
        Devolve todos os documentos cujo campo 'autores' contém nome_autor
        (pesquisa case-insensitive por substring).
        """
        nome_lower = nome_autor.lower()
        resultados = []
        for doc_id, meta in self.documentos.items():
            for autor in meta.get("autores", []):
                if nome_lower in autor.lower():
                    resultados.append({
                        "doc_id":  doc_id,
                        "titulo":  meta.get("titulo", ""),
                        "autores": meta.get("autores", []),
                        "ano":     meta.get("ano", ""),
                        "url":     meta.get("url", ""),
                    })
                    break   # evitar duplicados se o mesmo autor aparecer duas vezes
        return sorted(resultados, key=lambda x: x.get("ano", ""), reverse=True)

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


# ══════════════════════════════════════════════════════════════════════════════
#  Exemplo de uso
# ══════════════════════════════════════════════════════════════════════════════

#if __name__ == "__main__":
    # ── Opção A: a partir do scraper_results.json ──────────────────────────
    # with open("scraper_results.json", encoding="utf-8") as f:
    #     scraper_results = json.load(f)
    #
    # idx = IndiceInvertido()
    # idx.construir_de_scraper(scraper_results, pasta_textos="textos_processados")
    # idx.guardar("indice_invertido.json")

    # ── Opção B: a partir do processed_corpus.json ─────────────────────────
    # with open("processed_corpus.json", encoding="utf-8") as f:
    #     corpus = json.load(f)
    #
    # idx = IndiceInvertido()
    # idx.construir_de_processed_corpus(corpus, pasta_textos="textos_processados")
    # idx.guardar("indice_invertido.json")

    # ── Carregar índice já existente e pesquisar ───────────────────────────
    # idx = IndiceInvertido()
    # idx.carregar("indice_invertido.json")
    #
    # # Pesquisa booleana
    # print(idx.pesquisa_booleana("machine AND learning"))
    # print(idx.pesquisa_booleana("python OR java"))
    # print(idx.pesquisa_booleana("NOT python"))
    # print(idx.pesquisa_booleana("(neural OR deep) AND learning"))
    #
    # # Ranking TF-IDF
    # for r in idx.pesquisa_tfidf("deep learning neural network", top_k=5):
    #     print(f"  {r['score']:.4f}  {r['titulo']}")
    #
    # # Por autor
    # for r in idx.pesquisa_por_autor("Santos"):
    #     print(r["titulo"])
    #
    # print(idx.estatisticas())

    #print("indice.py importado com sucesso. Use IndiceInvertido() para começar.")