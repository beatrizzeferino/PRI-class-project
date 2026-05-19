import os
from src.search.nlp import TextProcessor

class ModeloBooleano:
    def __init__(self, corpus_processado, pasta_tokens_pdf, remove_stopwords, normalization_method, language):
        self.corpus = corpus_processado  #json processed_corpus.json

        self.pasta_tokens_pdf = pasta_tokens_pdf #pasta onde estao os pdfs processados

        self.termos_unicos = []   # lista ordenada de termos (linhas)
        self.doc_ids = []         # lista dos doc_ids (colunas)
        self.matriz = []          # matriz termos-documentos

        self.termo_indice = {}

        self.remove_stopwords = remove_stopwords
        self.normalization_method = normalization_method
        self.language = language

        # menor valor -> maior prioridade
        self.prioridade = {
            "(": 0,
            ")": 0,
            "NOT": 1,
            "AND": 2,
            "OR": 3
        }

        self.nlp = TextProcessor()

    def carregar_tokens_documento(self, doc_id, info_doc):
        """
        Junta os tokens do scraper + tokens do pdf processado de um determinado documento
        """

        tokens = list(info_doc.get("tokens_pesquisa", []))

        doc_id_pdf = str(doc_id).replace("/", "_").replace("\\", "_")
        nome_ficheiro = f"{doc_id_pdf}_tokens.txt"

        caminho_pdf= os.path.join(self.pasta_tokens_pdf, nome_ficheiro)

        if os.path.exists(caminho_pdf):
            try:
                with open(caminho_pdf, "r", encoding="utf-8") as f:
                    tokens_pdf = f.read().strip().split()

                    tokens.extend(tokens_pdf)
            except Exception as e:
                print(f"[Erro PDF Tokens] {doc_id}: {e}")

        return tokens

    def construir_matriz(self):
        """
        Constrói a matriz termo-documento: lista de listas onde cada lista
        interna indica se um termo existe (1) ou não (0) em cada documento.
        """
        termos = set()
        docs_tokens = []
        self.doc_ids = []
        self.termo_indice = {}

        for doc_id, doc in self.corpus.items():
            self.doc_ids.append(doc_id)

            tokens = self.carregar_tokens_documento(doc_id, doc)
            tokens_set = set(tokens)

            docs_tokens.append(tokens_set)
            termos.update(tokens_set)

        self.termos_unicos = sorted(list(termos))

        for i, termo in enumerate(self.termos_unicos):
            self.termo_indice[termo] = i

        num_docs = len(self.doc_ids)
        num_termos = len(self.termos_unicos)

        # inicializar matriz
        self.matriz = [[0] * num_docs for _ in range(num_termos)]

        # preencher a matriz
        for doc_indice, doc_tokens in enumerate(docs_tokens):
            for termo in doc_tokens:
                termo_indice = self.termo_indice[termo]
                self.matriz[termo_indice][doc_indice] = 1

        # BUGFIX: era self.documentos (não existe) — correto é self.doc_ids
        print(f"[Modelo Booleano] Matriz termo-documento construída: {len(self.termos_unicos)} termos x {len(self.doc_ids)} documentos.")


# =========== Resoluções de Queries ===============================================

    def obter_linha_termo(self, termo):
        """Devolve a linha (vetor) correspondente ao termo pesquisado; se não existir, devolve zeros."""
        tokens = self.nlp.process_text(
            termo,
            language=self.language,
            remove_stopwords=self.remove_stopwords,
            normalization_method=self.normalization_method
        )

        if not tokens:
            return [0] * len(self.doc_ids)

        termo_proc = tokens[0]

        indice = self.termo_indice.get(termo_proc)

        if indice is not None:
            return self.matriz[indice]

        return [0] * len(self.doc_ids)

    def operacao_and(self, linha_termo1, linha_termo2):
        """Devolve vetor AND (interseção) entre dois vetores."""
        return [a & b for a, b in zip(linha_termo1, linha_termo2)]

    def operacao_and_otimizado(self, lista_vetores):
        """AND otimizado: ordena pelo número de 1s (termos raros primeiro)."""
        vetores_ordenados = sorted(lista_vetores, key=lambda x: sum(x))

        resultado = vetores_ordenados[0]
        for i in range(1, len(vetores_ordenados)):
            resultado = self.operacao_and(resultado, vetores_ordenados[i])
        return resultado

    def operacao_or(self, linha_termo1, linha_termo2):
        """Devolve vetor OR (união) entre dois vetores."""
        return [a | b for a, b in zip(linha_termo1, linha_termo2)]

    def operacao_not(self, linha):
        """Inverte o vetor (complemento)."""
        return [1 if x == 0 else 0 for x in linha]

    def avaliar_query(self, query):
        """
        Resolve a query respeitando a hierarquia dos operadores lógicos
        e gere AND implícito entre termos separados por espaço.
        """
        query = query.replace("(", " ( ").replace(")", " ) ")
        tokens_brutos = query.split()

        tokens = []

        for i in range(len(tokens_brutos)):
            token_atual = tokens_brutos[i]
            tokens.append(token_atual)

            if i < len(tokens_brutos) - 1:
                proximo = tokens_brutos[i + 1]

                atual_e_termo = token_atual not in self.prioridade
                proximo_e_termo = proximo not in self.prioridade

                if atual_e_termo and proximo_e_termo:
                    tokens.append("AND")
                elif atual_e_termo and proximo == "NOT":
                    tokens.append("AND")
                elif atual_e_termo and proximo == "(":
                    tokens.append("AND")
                elif token_atual == ")" and proximo_e_termo:
                    tokens.append("AND")

        operadores = []
        vetores = []

        def resolver_ultimo():
            if not operadores:
                return None

            op = operadores.pop()
            if op == "NOT":
                val = vetores.pop()
                vetores.append(self.operacao_not(val))
            elif op == "AND":
                lista_para_and = [vetores.pop(), vetores.pop()]
                while operadores and operadores[-1] == "AND":
                    operadores.pop()
                    lista_para_and.append(vetores.pop())
                vetores.append(self.operacao_and_otimizado(lista_para_and))
            elif op == "OR":
                if len(vetores) >= 2:
                    dir_ = vetores.pop()
                    esq = vetores.pop()
                    vetores.append(self.operacao_or(esq, dir_))

        for token in tokens:
            if token == "(":
                operadores.append(token)
            elif token == ")":
                while operadores and operadores[-1] != "(":
                    resolver_ultimo()
                operadores.pop()  # remove "("
            elif token in self.prioridade:
                while (operadores and operadores[-1] != "(" and
                       self.prioridade[operadores[-1]] <= self.prioridade[token]):
                    resolver_ultimo()
                operadores.append(token)
            else:
                vetores.append(self.obter_linha_termo(token))

        while operadores:
            resolver_ultimo()

        if not vetores:
            return [0] * len(self.doc_ids)

        return vetores[0]

    def executar_pesquisa(self, query):
        """
        Executa a query e devolve uma lista de doc_ids dos documentos encontrados.
        (O app.py faz o mapeamento para os documentos completos via db_documentos.)
        """
        resultado_binario = self.avaliar_query(query)

        doc_ids_res = []
        for i, bit in enumerate(resultado_binario):
            if bit == 1:
                doc_ids_res.append(self.doc_ids[i])

        return doc_ids_res