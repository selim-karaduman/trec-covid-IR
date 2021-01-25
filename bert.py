from sentence_transformers import SentenceTransformer
from base import *
from joblib import dump, load

"""
This assumes a baseline has been trained:
    if not train either svd or tf-idf model e.g. for tf-idf:
        b = TfIdfBaseline()
        b.extract_stats_to_file(corpus, cosfname)

Extract the embedding over the complete corpus once, to do this run:
l = BertRanker(fname_base)
l.extract_stats_to_file(fname)

# then to use this to extract documents:
To load existing statistics:
    l = BertRanker(fname_base)
    l.load(fname)
    l.get_ranked_documents(query, alpha)

"""
class BertRanker(TfIdfBaseline):

    def __init__(self, fname_base):
        super().__init__()
        print("Loading baseline...")
        super().load(fname_base)
        self.model = SentenceTransformer("./models/covidbert-nli")

    def extract_stats_to_file(self, corpus, queries, fname):

        self.process_queries(queries, False)
        self.process_corpus(corpus, False)
        self.save(fname + "notclean")

        self.process_queries(queries, True)
        self.process_corpus(corpus, True)
        self.save(fname + "clean")

    def save(self, fname):
        # query_embeddings: is a dictionary: 
        #    "query string" -> bert embedding
        # doc_embeddings: is an np.array; embedding of doc with id i:
        #    self.doc_embeddings[i, :]
        dump((self.query_embeddings, self.doc_embeddings), fname)

    def load(self, fname):
        self.query_embeddings, self.doc_embeddings = load(fname)
        

    def get_ranked_docs(self, query, alpha=0.2, k=-1):
        """
        query: str
        alpha: int: interpolation parameter, final ranking is:
            base_score * (1-alpha) + bert_score * (alpha)
        """
        if query in self.query_embeddings:
            query_embedding = self.query_embeddings[query]
        else:
            query = self.clean_string(query)
            query_embedding = self.model.encode([query])
        query_embedding = query_embedding.reshape(1, -1)
        # query_embedding: (1, E)
        # self.doc_embeddings: (N, E)
        bert_sim_matrix = cosine_similarity(self.doc_embeddings,
                                         query_embedding)[:, 0]
        base_ranked_docs = super().get_ranked_docs(query, k=k)
        cord_uid2id = {c_i:i for i, c_i in enumerate(self.doc_ids)}

        ranked_docs = []
        for base_score, cord_uid in base_ranked_docs:
            bert_score = bert_sim_matrix[cord_uid2id[cord_uid]]
            final_score = (1-alpha)*base_score + alpha*bert_score
            ranked_docs.append([final_score, cord_uid])
        
        return sorted(ranked_docs, key=lambda x: -x[0])

    def clean_string(self, text):
        return " ".join(super().process_text(text))

    def process_corpus(self, corpus, clean):
        # index bt cord_uid
        cord_dict = {d_id: i for i, d_id in enumerate(self.doc_ids)}
        corpus_d = set()
        doc_texts = [None]*len(self.doc_ids)
        print("Distilling articles")
        for i in range(len(corpus)):
            if ((corpus["cord_uid"][i] in cord_dict) and 
                    (corpus["cord_uid"][i] not in corpus_d)):
                cord_uid = corpus["cord_uid"][i]
                corpus_d.add(cord_uid)
                d = corpus.iloc[i]
                title = d["title"]
                title = "" if (not isinstance(title, str)) else title
                abstract = d["abstract"]
                abstract = "" if (not isinstance(abstract, str)) else abstract
                text= title + " " + abstract
                if clean:
                    text = self.clean_string(text)
                doc_texts[cord_dict[cord_uid]] = text
            if i % 1000 == 0:
                print("{}/{} ; {:.2f} %".format(i, len(corpus),
                                                     i/len(corpus)*100))

        
        print("Calculating embeddings")
        embeddings = self.model.encode(doc_texts, show_progress_bar=True)
        self.doc_embeddings = np.stack(embeddings)

    def process_queries(self, queries, clean):
        print("Processing queries")
        self.query_embeddings = dict()
        # clean up the queries
        if clean:
            queries = [self.clean_string(q) for q in queries]
        embeddings = self.model.encode(queries, show_progress_bar=True)
        for i in range(len(queries)):
            self.query_embeddings[queries[i]] = embeddings[i]


