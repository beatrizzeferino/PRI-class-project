import math
from src.search.nlp import TextProcessor

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDF:
    def __init__(self, indice, documentos, tf_scheme, idf_scheme, remove_stopwords, normalization_method, language):
        self.indice = indice
        self.documentos= documentos # output do CorpusProcessor
        self.N= indice.num_documentos #numero total de documentos no corpus

        #weighting schemes escolhidos pelo utilizador
        self.tf_scheme= tf_scheme
        self.idf_scheme= idf_scheme

        #necessario para processar o texto da query pelo nlp
        self.remove_stopwords = remove_stopwords
        self.normalization_method = normalization_method
        self.language = language

        self.nlp = TextProcessor()
    
    def calcular_tf_score(self, tf):
        """
        Calcula o peso TF de um termo num documento, atraves de diferentes schemes
        """
        if tf == 0:
            return 0
        if self.tf_scheme == "raw":
            return tf
        elif self.tf_scheme == "binary":
            return 1
        elif self.tf_scheme== "log":
            return 1 + math.log(tf,10)
        elif self.tf_scheme == "augmented":
            return 0.5 +0.5 * (tf/ (tf+1))
        
        return 1 + math.log(tf,10) # caso nao corresponder a nenhuma das opcoes assumimos o logaritmo pois é a mais usualmente usada
    
    def calcular_tf_termo(self, termo):
        """
        Calcula o TF para um termo em todos os documentos
        Devolve um dicionário {doc_id : tf_ponderado}
        """
        posting_list = self.indice.obter_posting_list(termo)
        
        if not posting_list:
            return {}
        
        tf_scores={}

        #percorre todos os documentos onde o termo aparece
        for posting in posting_list.postings:
            doc_id = posting["doc_id"]
            tf = posting["tf"]

            #aplica o tf score escolhido
            tf_scores[doc_id] = self.calcular_tf_score(tf)

        return tf_scores
    
    def calcular_idf(self, termo):
        """
        Calcula o IDF de um termo
        """
        posting_list = self.indice.obter_posting_list(termo)

        if not posting_list:
            return 0
        
        df = posting_list.df #document frequency
        
        #para evitar casos onde se dividiria por 0, para nao rebentar
        if df== 0:
            return 0
        
        #diferentes schemes que o utilizador escolheu
        if self.idf_scheme == "standard":
            return math.log((self.N) / (df), 10) 
        
        elif self.idf_scheme == "smooth":
            return math.log((1 + self.N) / (1 + df),10)
        
        elif self.idf_scheme == "probabilistic":
            if df == self.N:
                return 0
            return max(0,math.log((self.N - df)/ df, 10)) #para garantir que nao retorna scores negativos caso df > N/2
        
        return math.log((self.N) / (df), 10)  #caso nao corresponda a nenhuma das opcoes retorna a standard pois normalmente é a mais usada
    

    def calcular_tfidf_termo(self, termo):
        """
        Calcula o TF-IDF de um termo para todos os documentos
        Devolve {doc_id: tf-idf} para um termo especifico
        """

        posting_list= self.indice.obter_posting_list(termo)
        if not posting_list:
            return{}
        
        idf = self.calcular_idf(termo)
        
        tfidf_scores = {}

        for posting in posting_list.postings:
            doc_id = posting["doc_id"]
            tf = posting["tf"]

            tf_score = self.calcular_tf_score(tf)

            #TF-IDF = TF * IDF
            tfidf_scores[doc_id] = tf_score * idf

        return tfidf_scores
    

    #-------------------Similaridade----------------------------
    def processar_query(self, query):
        """
        Aplica o mesmo processamento NLP usado nos documentos à query
        """

        return self.nlp.process_text(
            query,
            language=self.language,
            remove_stopwords=self.remove_stopwords,
            normalization_method=self.normalization_method
        )

    def vetor_tfidf_query(self, query_tokens):
        """
        Construção do vetor TF-IDF da query
        Devolve {termo: peso TF-IDF na query}
        """

        vetor={}

        tf_query = {}

        #calculo do tf da query
        for termo in query_tokens:
            tf_query[termo] = tf_query.get(termo, 0) + 1

        #construcao do vetor
        for termo, tf in tf_query.items():
            tf_score = self.calcular_tf_score(tf)
            idf = self.calcular_idf(termo)

            vetor[termo] = tf_score * idf
        
        return vetor
    
    def vetor_tfidf_documento(self, doc_id, termos):
        """
        Constrói o vetor TF-IDF de um documento em específico
        """
        vetor= {}

        for termo in termos:
            tfidf_dic = self.calcular_tfidf_termo(termo)

            if doc_id in tfidf_dic:
                vetor[termo] = tfidf_dic[doc_id]
        
        return vetor
    
    def similaridade_cosseno(self, vec_doc, vec_q):
        """
        Calcula a similaridade de cosseno entre um vetor de documento e de query
        """
        produto= 0

        #produto escalar
        for termo in vec_q:
            if termo in vec_doc:
                produto += vec_doc[termo] * vec_q[termo]

        norma_doc = math.sqrt(sum(v**2 for v in vec_doc.values()))
        norma_q = math.sqrt(sum(v**2 for v in vec_q.values()))
        
        #evitar divisao por 0
        if norma_doc == 0 or norma_q == 0:
            return 0
        
        return produto / (norma_doc * norma_q)
    
    def rank_documentos(self, query):
        """
        Retorna documentos ordenados por relevância à query
        """
        query_tokens = self.processar_query(query)

        query_vec= self.vetor_tfidf_query(query_tokens)

        scores= {}

        for doc_id in self.indice.documentos:
            doc_vec = self.vetor_tfidf_documento(doc_id, query_tokens)

            score= self.similaridade_cosseno(doc_vec, query_vec)

            if score > 0:
                scores[doc_id] = score

        #ordenacao decrescente por score
        return sorted(scores.items(), key= lambda x: x[1], reverse=True) #retorna uma lsita de tuplos [(doc_id, score), (doc_id, score)]

    def gerar_matriz_similaridade(self):
        """
        Gerar matriz N*N de similaridade entre documentos (REQ-B40)
        Cada celula (i, j) representa a similaridade entre dois documentos
        """
        docs = list(self.indice.documentos.keys())
        matriz={}

        #pre-calcular vetores de todos os documentos
        vetores={} #vetores de todos os documentos

        for doc_id in docs:
            termos= self.documentos[doc_id]["tokens_pesquisa"]
            vetores[doc_id] = self.vetor_tfidf_documento(doc_id, termos)
        
        #calcular similaridades
        for i, doc_i in enumerate(docs):
            matriz[doc_i] ={}

            for j, doc_j in enumerate(docs):
                if j <i:
                    #reaproveitar a simetria da matriz
                    matriz[doc_i][doc_j] = matriz[doc_j][doc_i]
                
                if doc_i == doc_j:
                    matriz[doc_i][doc_j] = 1.0

                else:
                    sim = self.similaridade_cosseno(vetores[doc_i], vetores[doc_j])
                    matriz[doc_i][doc_j] = sim
        return matriz

class TFIDF_Sklearn():
    #TF-IDF mas utilizando a biblioteca sklearn
    def __init__(self, documentos_processados, remove_stopwords, normalization_method, language):
        #sklearn trabalha diretamente sobre textos
        self.documentos = documentos_processados #output do CorpusProcessor

        self.vectorizer= TfidfVectorizer() #objeto do sk-learn que sabe construir o vocabulario, calcular tf e idf e gerar vetores tf-idf

        corpus = self.preparar_corpus()
        self.matriz_tfidf = self.vectorizer.fit_transform(corpus) #fit aprende o vocabulario, transform transforma documentos em vetores numericos tf-idf, isto leva a uma matriz do tipo n_documento x n_termos

        self.nlp = TextProcessor()
        self.remove_stopwords= remove_stopwords
        self.normalization_method = normalization_method
        self.language = language

    def preparar_corpus(self):
        '''
        Converte os documentos processados num formato compativel com o sklearn
        '''
        
        #sklearn espera string completas não listas de strings por documento
        
        corpus=[]
        self.doc_ids= []

        for doc_id, info in self.documentos.items():

            texto= " ".join(info["tokens_pesquisa"])

            corpus.append(texto)
            self.doc_ids.append(doc_id)
        
        return corpus

    def rank_documentos(self,query):
        '''
        Executa a pesquisa de documentos relevantes para uma query
        Retorna uma lista de tuplos (doc_id, score) ordenada por relevancia decrescente
        '''
        query_tokens = self.nlp.process_text(
            query,
            language=self.language,
            remove_stopwords=self.remove_stopwords,
            normalization_method=self.normalization_method
        )   

        query_texto = " ".join(query_tokens)

        query_vec = self.vectorizer.transform([query_texto]) # nao aprende novo vocabulario nem recalcula IDF mas sim transforma a query usando o vocabulario já aprendido nos documentos e os idfs para calcular o vetor da query

        scores = cosine_similarity(query_vec, self.matriz_tfidf)[0] #cosine_similarity da uma lista de lista e queremos apenas a lista interior

        ranking = sorted(
            zip(self.doc_ids, scores), #junta o doc_id e o seu respetivo score
            key=lambda x: x[1],
            reverse=True
        )

        #remover scores 0
        ranking = [
            (doc_id, score)
            for doc_id, score in ranking
            if score > 0
        ]

        return ranking