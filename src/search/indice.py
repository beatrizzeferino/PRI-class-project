import json
import math


class PostingList:
    def __init__(self):
        self.postings = []       # Lista de {"doc_id": ..., "tf": ...}
        self.skip_pointers = {}  # {indice_i: indice_j} — saltos para acelerar interseção
        self.df = 0              # Document frequency (quantos docs têm este termo)

    def adicionar_posting(self, doc_id, tf):
        """Adiciona ou atualiza o posting de um documento."""
        for posting in self.postings:
            if posting["doc_id"] == doc_id:
                posting["tf"] = tf
                return
        self.postings.append({"doc_id": doc_id, "tf": tf})
        self.df = len(self.postings)

    def ordenar(self):
        """Ordena os postings por doc_id (obrigatório para os skip pointers e interseções)."""
        self.postings.sort(key=lambda x: x["doc_id"])

    def construir_skip_pointers(self):
        """
        Constrói skip pointers com salto de sqrt(n).
        Cada posição i aponta para i + sqrt(n), permitindo saltar entradas
        durante a interseção quando o doc_id atual é menor que o do outro lado.
        """
        self.skip_pointers = {}
        n = len(self.postings)
        if n <= 3:
            return
        salto = int(math.sqrt(n))
        for i in range(0, n - salto, salto):
            self.skip_pointers[i] = i + salto

    def doc_ids(self):
        """Devolve apenas a lista de doc_ids (útil para debug e para o modelo booleano)."""
        return [p["doc_id"] for p in self.postings]


class IndiceInvertido:
    def __init__(self):
        self.indice = {}       # termo -> PostingList
        self.documentos = {}   # doc_id -> metadados
        self.num_documentos = 0

    # ------------------------------------------------------------------ #
    #  Construção                                                          #
    # ------------------------------------------------------------------ #

    def construir_de_indexer(self, documentos_processados):
        """
        Constrói o índice a partir do dicionário devolvido pelo Indexer.
        Chama _indexar_documento para cada doc e no final finaliza o índice.
        """
        for doc_id, info in documentos_processados.items():
            self._indexar_documento(doc_id, info)
        self._finalizar_indice()

    def _indexar_documento(self, doc_id, info):
        """
        Indexa um único documento:
        1. Guarda os metadados em self.documentos
        2. Conta as ocorrências de cada token (TF bruto)
        3. Adiciona o posting em cada PostingList do índice
        """
        # 1. Metadados
        self.documentos[doc_id] = {
            "titulo":  info.get("titulo", ""),
            "url":     info.get("url", ""),
            "autores": info.get("autores", []),
            "ano":     info.get("ano", ""),
            "idioma":  info.get("idioma", "english"),
        }

        # 2. Contar TF por token
        tokens = info.get("tokens_pesquisa", [])
        tf_doc = {}
        for token in tokens:
            t = token.lower()
            tf_doc[t] = tf_doc.get(t, 0) + 1

        # 3. Atualizar índice
        for termo, tf in tf_doc.items():
            if termo not in self.indice:
                self.indice[termo] = PostingList()
            self.indice[termo].adicionar_posting(doc_id, tf)

        self.num_documentos = len(self.documentos)

    def _finalizar_indice(self):
        """Ordena todas as posting lists e constrói os skip pointers."""
        for pl in self.indice.values():
            pl.ordenar()
            pl.construir_skip_pointers()

    # ------------------------------------------------------------------ #
    #  Atualização incremental                                             #
    # ------------------------------------------------------------------ #

    def adicionar_documentos(self, novos_documentos_processados):
        """
        Adiciona novos documentos a um índice já existente sem o reconstruir
        de raiz. Após inserir todos os novos docs, re-ordena e reconstrói os
        skip pointers apenas nas posting lists afetadas.

        Parâmetro:
            novos_documentos_processados (dict): mesmo formato que o devolvido
            pelo Indexer — {doc_id: {tokens_pesquisa, titulo, url, ...}}
        """
        termos_afetados = set()

        for doc_id, info in novos_documentos_processados.items():
            if doc_id in self.documentos:
                print(f"[Aviso] Documento '{doc_id}' já existe no índice. A ignorar.")
                continue

            # Indexar o documento e registar quais termos foram tocados
            self._indexar_documento(doc_id, info)

            tokens = info.get("tokens_pesquisa", [])
            termos_afetados.update(t.lower() for t in tokens)

        # Re-ordenar e reconstruir skip pointers apenas nas listas afetadas
        for termo in termos_afetados:
            if termo in self.indice:
                self.indice[termo].ordenar()
                self.indice[termo].construir_skip_pointers()

        print(f"[Índice] {len(novos_documentos_processados)} documento(s) adicionado(s). "
              f"Total: {self.num_documentos}")

    # ------------------------------------------------------------------ #
    #  Pesquisa                                                            #
    # ------------------------------------------------------------------ #

    def obter_posting_list(self, termo):
        """Devolve a PostingList de um termo (ou None se não existir)."""
        return self.indice.get(termo.lower())

    def intersetar_com_skip(self, lista1, lista2):
        """
        Interseção AND de duas PostingLists com skip pointers.

        Os skip pointers permitem saltar blocos da lista quando o doc_id
        corrente é menor que o do outro lado — reduz comparações de O(m+n)
        para O(m+n) no pior caso mas muito menos na prática.

        O TF resultante é a soma dos TFs (convenção IR: AND implica que o
        termo aparece em ambos, por isso acumulamos a evidência).
        """
        resultado = PostingList()
        i, j = 0, 0
        p1, p2 = lista1.postings, lista2.postings

        while i < len(p1) and j < len(p2):
            if p1[i]["doc_id"] == p2[j]["doc_id"]:
                # Documento em comum — TF combinado = soma dos dois TFs
                tf_combinado = p1[i]["tf"] + p2[j]["tf"]
                resultado.adicionar_posting(p1[i]["doc_id"], tf_combinado)
                i += 1
                j += 1
            elif p1[i]["doc_id"] < p2[j]["doc_id"]:
                # Tentar saltar em lista1
                if (i in lista1.skip_pointers and
                        p1[lista1.skip_pointers[i]]["doc_id"] <= p2[j]["doc_id"]):
                    i = lista1.skip_pointers[i]
                else:
                    i += 1
            else:
                # Tentar saltar em lista2
                if (j in lista2.skip_pointers and
                        p2[lista2.skip_pointers[j]]["doc_id"] <= p1[i]["doc_id"]):
                    j = lista2.skip_pointers[j]
                else:
                    j += 1

        return resultado

    def unir(self, lista1, lista2):
        """
        União OR de duas PostingLists (merge linear, sem necessidade de skip pointers).
        Útil para pesquisas booleanas OR no modelo de índice invertido.
        """
        resultado = PostingList()
        i, j = 0, 0
        p1, p2 = lista1.postings, lista2.postings

        while i < len(p1) and j < len(p2):
            if p1[i]["doc_id"] == p2[j]["doc_id"]:
                resultado.adicionar_posting(p1[i]["doc_id"], p1[i]["tf"] + p2[j]["tf"])
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

    # ------------------------------------------------------------------ #
    #  Estatísticas                                                        #
    # ------------------------------------------------------------------ #

    def estatisticas(self):
        """Devolve um dicionário com estatísticas do índice."""
        return {
            "num_documentos": self.num_documentos,
            "num_termos_unicos": len(self.indice),
            "top_10_termos_por_df": sorted(
                [(t, pl.df) for t, pl in self.indice.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

    # ------------------------------------------------------------------ #
    #  Persistência                                                        #
    # ------------------------------------------------------------------ #

    def guardar(self, caminho):
        """
        Serializa o índice e os metadados dos documentos para JSON.
        CORREÇÃO: a versão anterior não guardava self.documentos,
        o que causava KeyError ao carregar.
        """
        dados_finais = {
            "num_documentos": self.num_documentos,
            "documentos": self.documentos,   # <-- CORREÇÃO: estava em falta
            "indice": {
                termo: {
                    "df":       pl.df,
                    "postings": pl.postings
                }
                for termo, pl in self.indice.items()
            }
        }
        with open(caminho, 'w', encoding='utf-8') as f:
            json.dump(dados_finais, f, ensure_ascii=False, indent=2)
        print(f"[OK] Índice guardado em: {caminho}")

    def carregar(self, caminho):
        """
        Carrega o índice de um ficheiro JSON previamente guardado com guardar().
        Reconstrói os skip pointers (não são serializados).
        """
        with open(caminho, 'r', encoding='utf-8') as f:
            dados = json.load(f)

        self.num_documentos = dados["num_documentos"]
        self.documentos     = dados.get("documentos", {})  # <-- CORREÇÃO: chave correta
        self.indice = {}

        for termo, info in dados["indice"].items():
            pl = PostingList()
            pl.postings = info["postings"]
            pl.df       = info["df"]
            pl.construir_skip_pointers()
            self.indice[termo] = pl

        print(f"[OK] Índice carregado: {self.num_documentos} documentos, "
              f"{len(self.indice)} termos.")


# ------------------------------------------------------------------ #
#  Bloco principal — exemplo de uso                                    #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    from src.search.indexer import Indexer

    # 1. Processar documentos com o Indexer
    indexer = Indexer()
    documentos_processados = indexer.processar_dataset(
        "scraper_results.json",
        remove_stopwords=True,
        normalization_method='lemma'
    )

    # 2. Construir o índice
    indice = IndiceInvertido()
    indice.construir_de_indexer(documentos_processados)

    # 3. Estatísticas
    stats = indice.estatisticas()
    print(f"\nEstatísticas do índice:")
    print(f"  Documentos    : {stats['num_documentos']}")
    print(f"  Termos únicos : {stats['num_termos_unicos']}")
    print(f"  Top 10 termos : {stats['top_10_termos_por_df']}")

    # 4. Exemplo de interseção com skip pointers
    pl_use    = indice.obter_posting_list("use")
    pl_system = indice.obter_posting_list("system")

    if pl_use and pl_system:
        resultado = indice.intersetar_com_skip(pl_use, pl_system)
        print(f"\nResultados 'use AND system': {len(resultado.postings)} documentos")
        for p in resultado.postings[:5]:
            meta = indice.documentos.get(p["doc_id"], {})
            print(f"  - {meta.get('titulo', p['doc_id'])} (tf={p['tf']})")

    # 5. Guardar índice no disco
    indice.guardar("indice_invertido.json")

    # 6. Demonstração de atualização incremental
    # (simulação — na prática viria do scraper)
    novos_docs = indexer.processar_dataset(
        "scraper_results.json",
        remove_stopwords=True,
        normalization_method='lemma'
    )
    indice.adicionar_documentos(novos_docs)
    indice.guardar("tests/indice_invertido.json")  # re-guardar com os novos docs