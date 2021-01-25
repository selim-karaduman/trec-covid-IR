import numpy as np
import scipy
from document import Document
import pickle
from base import Base
from sklearn.preprocessing import normalize


class BM25Baseline(Base):
    def __init__(self, b=0.75, k=1.5):
        super().__init__()
        self.b = b
        self.k = k
        
    def extract_stats_to_file(self, corpus, fname):
        self.process_corpus(corpus)
        self.save(fname)

    def save(self, fname):
        print("Calculations are done, saving to disk")
        t = (self.idf, self.posting_list, self.bm25_matrix, 
                self.doc_ids, self.word2id, self.d_avg, self.b, self.k)
        with open(fname, "wb") as f:
            pickle.dump(t, f)

    def load(self, fname):
        with open(fname, "rb") as f:
            t = pickle.load(f)
        (self.idf, self.posting_list, self.bm25_matrix, 
            self.doc_ids, self.word2id, self.d_avg, self.b, self.k) = t
  
    def get_ranked_docs(self, query, k=-1):
        """
        -- this assumes that no mutual documents in the self.doc_ids
        returns np.array of shape (N,)
        k: number of retrievals: k=-1 returns all
        """
        sim_matrix, sim_id2doc_id = self.get_sim_matrix(query, k)
        return self.get_sorted_docs(sim_matrix, sim_id2doc_id, k)

    def get_sim_matrix(self, query, k):
        tokenized_text = self.process_text(query)
        query_doc = Document(tokenized_text)
        #query_doc.cache_bm25_vector(self.word2id, self.b, self.k, self.d_avg)
        q_tf = query_doc.get_tf_vec(self.word2id) # (1, V)
        q_bm = q_tf * self.bm25_matrix.T
        # divide by max to get 0-1 range
        q_bm = q_bm/q_bm.max()
        return q_bm, None

    ##def get_sim_matrix(self, query, k):
    ##    q_v = self.encode_query(query)
    ##    return super().calculate_sim_matrix(self.bm25_matrix, q_v, k)
   
    def process_corpus(self, corpus):
        idf, docs = super().process_corpus(corpus)
        print("Processed files. Now getting bm25 statistics")
        N = len(docs)
        for i, doc in enumerate(docs):
            doc.cache_bm25_vector(self.word2id, self.b, self.k, self.d_avg)
            if i % 100 == 0:
                print("{} / {}; {:.2f} %".format(i, N, i / N * 100))
        idf = np.array(idf)
        idf = np.log(((N - idf + 0.5) / (idf + 0.5)) + 1)
        self.idf = scipy.sparse.diags(idf, format='csr')
        doc_bm = scipy.sparse.vstack([doc.bm_vec for doc in docs])
        self.bm25_matrix = doc_bm * self.idf

