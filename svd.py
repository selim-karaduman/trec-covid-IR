from tfidf import TfIdfBaseline
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import TruncatedSVD
import numpy as np
from joblib import dump, load
from document import *

"""
This assumes a TfIdfBaseline has been trained:
    if not train it with:
        b = TfIdfBaseline()
        b.extract_stats_to_file(corpus, cosfname)

To train:
l = SvdBaseline(cosfname)
l.extract_stats_to_file(fname)

To load existing statistics:
    l = SvdBaseline(cosfname)
    l.load(fname)
    ...

"""
class SvdBaseline(TfIdfBaseline):
    
    def __init__(self, fname_base):
        super().__init__()
        super().load(fname_base)

    def extract_stats_to_file(self, fname, n_iter=20, n=100):
        self.fit(n_iter, n)
        self.save(fname)

    def save(self, fname):
        dump((self.svd, self.doc_mat), fname)

    def load(self, fname):
        self.svd, self.doc_mat = load(fname)

    def get_ranked_docs(self, query, k=-1):
        sim_matrix, sim_id2doc_id = self.get_sim_vector(query, k)
        return self.get_sorted_docs(sim_matrix, sim_id2doc_id, k)

    def encode_query(self, tokenized_text):
        query_doc = Document(tokenized_text)
        query_doc.cache_tf_vector(self.word2id)
        q_tf_idf = query_doc.tf_vec * self.idf
        q_v = self.svd.transform(q_tf_idf)
        return q_v

    def get_sim_vector(self, query, k):
        tokenized_text = self.process_text(query)
        q_v = self.encode_query(tokenized_text)
        return super().calculate_sim_vector(self.doc_mat, 
                                                q_v, k, tokenized_text)

    def fit(self, n_iter, n):
        self.svd = TruncatedSVD(n_components=100, 
                                random_state=0, n_iter=n_iter)
        print("SVD: fitting...")
        self.svd.fit(self.tf_idf)
        self.doc_mat = self.svd.transform(self.tf_idf)

