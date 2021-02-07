import numpy as np
import scipy
from document import Document
import pickle
from base import Base
from sklearn.preprocessing import normalize

"""
To extract statistics etc. initially run:
    b = TfIdfBaseline()
    b.extract_stats_to_file(corpus, outfname)
To load existing statistics:
    b = TfIdfBaseline()
    b.load(fname)
"""
class TfIdfBaseline(Base):
    def __init__(self):
        super().__init__()
        
    def extract_stats_to_file(self, corpus, fname):
        self.process_corpus(corpus)
        self.save(fname)

    def save(self, fname):
        print("Calculations are done, saving to disk")
        t = (self.idf, self.posting_list, self.tf_idf, 
                self.doc_ids, self.word2id)
        with open(fname, "wb") as f:
            pickle.dump(t, f)

    def load(self, fname):
        with open(fname, "rb") as f:
            t = pickle.load(f)
        (self.idf, self.posting_list, self.tf_idf, 
            self.doc_ids, self.word2id) = t
  
    def get_ranked_docs(self, query, k=-1):
        """
        -- this assumes that no mutual documents in the self.doc_ids
        returns np.array of shape (N,)
        k: number of retrievals: k=-1 returns all
        """
        sim_matrix, sim_id2doc_id = self.get_sim_vector(query, k)
        return self.get_sorted_docs(sim_matrix, sim_id2doc_id, k)

    def encode_query(self, tokenized_text):
        query_doc = Document(tokenized_text)
        query_doc.cache_tf_vector(self.word2id)
        q_tf_idf = query_doc.tf_vec * self.idf
        return q_tf_idf

    def get_sim_vector(self, query, k):
        tokenized_text = self.process_text(query)
        q_v = self.encode_query(tokenized_text)
        return super().calculate_sim_vector(self.tf_idf, 
                                                q_v, k, tokenized_text)
   
    def process_corpus(self, corpus):
        idf, docs = super().process_corpus(corpus)
        print("Processed files. Now getting tf-idf statistics")
        V = len(idf)
        N = len(docs)
        for i, doc in enumerate(docs):
            doc.cache_tf_vector(self.word2id)
            if i % 100 == 0:
                print("{} / {}; {:.2f} %".format(i, N, i / N * 100))
        idf = np.array(idf) + 1
        idf = np.log((N + 1) / idf) + 1
        # idf: (V,);
        self.idf = scipy.sparse.diags(idf, format='csr')
        doc_tf = scipy.sparse.vstack([doc.tf_vec for doc in docs])
        # doc_tf: (N, V)
        # tf_idf: (N, V)
        self.tf_idf = doc_tf * self.idf
        # normalize tf_df
        normalize(self.tf_idf, norm='l2', axis=1, copy=False)
