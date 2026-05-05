import math
from collections import Counter
from src.search.nlp import TextProcessor


class TFIDFCustom:
    """
    Implementação própria de TF-IDF com similaridade do cosseno.

    Fórmula usada:
        tf(t,d)  = count(t,d) / len(d)           (frequência relativa)
        idf(t)   = log( N / (1 + df(t)) ) + 1    (suavizado para evitar divisão por zero)
        tfidf    = tf * idf
    """

    def __init__(self, remove_stopwords=True, normalization_method='lemma', language='english'):
        self.remove_stopwords = remove_stopwords
        self.normalization_method = normalization_method
        self.language = language

        self.nlp = TextProcessor()

        # Estruturas internas
        self.doc_ids = []           # lista ordenada de doc_ids
        self.vocab = []             # lista ordenada de termos do vocabulário
        self.vocab_index = {}       # termo -> índice na lista vocab
        self.tfidf_matrix = []      # lista de vetores TF-IDF, um por documento
        self.idf_values = {}        # termo -> valor IDF

    # ------------------------------------------------------------------
    #  Construção
    # ------------------------------------------------------------------

    def construir(self, corpus_processado: dict):
        """
        Recebe o dicionário devolvido pelo CorpusProcessor e constrói
        a matriz TF-IDF.

        Parâmetros:
            corpus_processado: {doc_id: {"tokens_pesquisa": [...], ...}}
        """
        self.doc_ids = []
        tokens_por_doc = []

        # 1. Recolher tokens de cada documento
        for doc_id, info in corpus_processado.items():
            self.doc_ids.append(doc_id)
            tokens_por_doc.append(info.get("tokens_pesquisa", []))

        num_docs = len(self.doc_ids)

        # 2. Construir vocabulário
        vocab_set = set()
        for tokens in tokens_por_doc:
            vocab_set.update(tokens)
        self.vocab = sorted(vocab_set)
        self.vocab_index = {t: i for i, t in enumerate(self.vocab)}

        # 3. Calcular DF (document frequency) por termo
        df = Counter()
        for tokens in tokens_por_doc:
            for t in set(tokens):
                df[t] += 1

        # 4. Calcular IDF suavizado
        self.idf_values = {
            t: math.log(num_docs / (1 + df[t])) + 1
            for t in self.vocab
        }

        # 5. Construir matriz TF-IDF
        self.tfidf_matrix = []
        for tokens in tokens_por_doc:
            tf_raw = Counter(tokens)
            total = len(tokens) if tokens else 1
            vec = [0.0] * len(self.vocab)
            for t, count in tf_raw.items():
                if t in self.vocab_index:
                    tf = count / total
                    idx = self.vocab_index[t]
                    vec[idx] = tf * self.idf_values[t]
            self.tfidf_matrix.append(vec)

        print(f"[TF-IDF Custom] Matriz construída: {num_docs} docs × {len(self.vocab)} termos.")

    # ------------------------------------------------------------------
    #  Pesquisa
    # ------------------------------------------------------------------

    def _vetorizar_query(self, query: str) -> list:
        """Converte a query num vetor TF-IDF usando o vocabulário existente."""
        tokens = self.nlp.process_text(
            query,
            language=self.language,
            remove_stopwords=self.remove_stopwords,
            normalization_method=self.normalization_method
        )
        tf_raw = Counter(tokens)
        total = len(tokens) if tokens else 1
        vec = [0.0] * len(self.vocab)
        for t, count in tf_raw.items():
            if t in self.vocab_index:
                tf = count / total
                idx = self.vocab_index[t]
                vec[idx] = tf * self.idf_values.get(t, 0.0)
        return vec

    @staticmethod
    def _cosine_similarity(vec_a: list, vec_b: list) -> float:
        """Calcula a similaridade do cosseno entre dois vetores."""
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def pesquisar(self, query: str, top_k: int = 50) -> list:
        """
        Pesquisa por similaridade do cosseno.

        Devolve uma lista de dicionários ordenados por score decrescente:
            [{"doc_id": ..., "score": ...}, ...]
        """
        if not query.strip():
            return []

        query_vec = self._vetorizar_query(query)

        # Se o vetor da query for todo zeros (termos desconhecidos), sem resultados
        if all(v == 0.0 for v in query_vec):
            return []

        scores = []
        for i, doc_vec in enumerate(self.tfidf_matrix):
            sim = self._cosine_similarity(query_vec, doc_vec)
            if sim > 0:
                scores.append({"doc_id": self.doc_ids[i], "score": round(sim, 6)})

        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_k]


# ----------------------------------------------------------------------

class TFIDFSklearn:
    """
    Wrapper em torno do TfidfVectorizer do scikit-learn.
    Útil para comparar com a implementação própria.
    """

    def __init__(self, remove_stopwords=True, normalization_method='lemma', language='english'):
        self.remove_stopwords = remove_stopwords
        self.normalization_method = normalization_method
        self.language = language

        self.nlp = TextProcessor()

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            self._TfidfVectorizer = TfidfVectorizer
            self._cosine_similarity = cosine_similarity
        except ImportError:
            raise ImportError("scikit-learn não está instalado. Execute: pip install scikit-learn")

        self.vectorizer = None
        self.tfidf_matrix = None
        self.doc_ids = []

    def construir(self, corpus_processado: dict):
        """
        Recebe o dicionário do CorpusProcessor e ajusta o TfidfVectorizer.
        Os tokens já pré-processados são reunidos em strings para o sklearn.
        """
        self.doc_ids = []
        corpus_strings = []

        for doc_id, info in corpus_processado.items():
            self.doc_ids.append(doc_id)
            tokens = info.get("tokens_pesquisa", [])
            corpus_strings.append(" ".join(tokens))

        # analyzer='word' — os tokens já estão pré-processados (não reaplicar NLP)
        self.vectorizer = self._TfidfVectorizer(analyzer='word', token_pattern=r'\S+')
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_strings)

        print(f"[TF-IDF sklearn] Matriz construída: {len(self.doc_ids)} docs × {self.tfidf_matrix.shape[1]} termos.")

    def pesquisar(self, query: str, top_k: int = 50) -> list:
        """
        Pesquisa por similaridade do cosseno usando o sklearn.

        Devolve lista de {"doc_id": ..., "score": ...} ordenada por score.
        """
        if not query.strip() or self.vectorizer is None:
            return []

        # Pré-processar a query da mesma forma que os documentos
        tokens = self.nlp.process_text(
            query,
            language=self.language,
            remove_stopwords=self.remove_stopwords,
            normalization_method=self.normalization_method
        )
        query_str = " ".join(tokens)

        query_vec = self.vectorizer.transform([query_str])
        sims = self._cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        resultados = [
            {"doc_id": self.doc_ids[i], "score": round(float(sims[i]), 6)}
            for i in range(len(self.doc_ids))
            if sims[i] > 0
        ]
        resultados.sort(key=lambda x: x["score"], reverse=True)
        return resultados[:top_k]