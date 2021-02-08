from sentence_transformers import SentenceTransformer
from base import *
from joblib import dump, load
import os.path
from os import path
"""
This assumes a baseline has been trained:
    if not train tf-idf model e.g. for tf-idf or any other baseline:
        b = TfIdfBaseline()
        b.extract_stats_to_file(corpus, cosfname)

Extract the embedding over the complete corpus once, to do this run:
l = BertRanker(b)
l.extract_stats_to_file(fname)

# then to use this to extract documents:
To load existing statistics:
    l = BertRanker(b)
    l.load(fname)
    l.get_ranked_documents(query)

Uses functions of base argument, 
    since the relevant things are not stored in Base
"""
class BertRanker(Base):

    def __init__(self, base, alpha=0.21):
        super().__init__()
        self.base = base
        if (path.exists("./models/covidbert-nli") and 
            path.isdir("./models/covidbert-nli")):
            self.model = SentenceTransformer("./models/covidbert-nli")
        else:
            print("Installing bert model...")
            self.model = SentenceTransformer("gsarti/covidbert-nli")
            self.model.save("./models/covidbert-nli")
        self.alpha = alpha

    def extract_stats_to_file(self, corpus, queries, fname):
        self.process_queries(queries, False)
        self.process_corpus(corpus, False)
        self.save(fname + "notclean")

    def save(self, fname):
        # query_embeddings: is a dictionary: 
        #    "query string" -> bert embedding
        # doc_embeddings: is an np.array; embedding of doc with id i:
        #    self.doc_embeddings[i, :]
        dump((self.query_embeddings, self.doc_embeddings), fname)

    def load(self, fname):
        self.query_embeddings, self.doc_embeddings = load(fname)
        
    def get_ranked_docs(self, query, k=-1):
        """
        query: str
        alpha: int: interpolation parameter, final ranking is:
            base_score * (1-alpha) + bert_score * (alpha)
        """
        alpha = self.alpha
        bert_sim_vector, sim_id2doc_id = self.get_sim_vector(query, k)
        base_sim_vector, _ = self.base.get_sim_vector(query, k)
        sim_vector = alpha * bert_sim_vector + (1 - alpha) * base_sim_vector
        return self.base.get_sorted_docs(sim_vector, sim_id2doc_id, k)
    
    def encode_query(self, query):
        if query in self.query_embeddings:
            query_embedding = self.query_embeddings[query]
        else:
            query = self.base.clean_string(query)
            query_embedding = self.model.encode([query])
        query_embedding = query_embedding.reshape(1, -1)
        return query_embedding

    def get_sim_vector(self, query, k):
        q_v = self.encode_query(query)
        tokenized_text = self.process_text(query)
        return self.base.calculate_sim_vector(self.doc_embeddings, 
                                                q_v, k, tokenized_text)
    
    def clean_string(self, text):
        return " ".join(super().process_text(text))

    def process_corpus(self, corpus, clean):
        # index bt cord_uid
        cord_dict = {d_id: i for i, d_id in enumerate(self.base.doc_ids)}
        corpus_d = set()
        doc_texts = [None]*len(self.base.doc_ids)
        print("Processing articles")
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
                    text = self.base.clean_string(text)
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
            queries = [self.base.clean_string(q) for q in queries]
        embeddings = self.model.encode(queries, show_progress_bar=True)
        for i in range(len(queries)):
            self.query_embeddings[queries[i]] = embeddings[i]
