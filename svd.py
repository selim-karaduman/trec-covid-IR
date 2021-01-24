from base import TfIdfBaseline
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

    def extract_stats_to_file(self, fname, n_iter=200, n=100):
        self.fit(n_iter, n)
        self.save(fname)

    def save(self, fname):
        dump((self.svd, self.doc_mat), fname)

    def load(self, fname):
        self.svd, self.doc_mat = load(fname)

    def get_ranked_docs(self, query, k=-1):
        tokenized_text = self.process_text(query)
        query_doc = Document(tokenized_text)
        query_doc.cache_tf_vector(self.word2id)
        q_tf_idf = query_doc.tf_vec * self.idf
        q_v = self.svd.transform(q_tf_idf)
        if k == -1:
            sim_matrix = cosine_similarity(self.tf_idf, q_v)[:, 0]
            doc_ids = (-sim_matrix).argsort()
            ranked_docs = []
            for doc_id in doc_ids:
                doc_corduid = self.doc_ids[doc_id]
                ranked_docs.append([sim_matrix[doc_id], doc_corduid])
            return ranked_docs
        else:
            cand_doc_ids = np.array(list
                            (set.union(
                                *[self.posting_list.get(word, set())
                                    for word in tokenized_text])
                            )
                        )
            sim_id2doc_id = {i:d_id for i, d_id in enumerate(cand_doc_ids)}
            sim_matrix = cosine_similarity(self.tf_idf[cand_doc_ids,:], 
                                            q_v)[:, 0]
            doc_ids = (-sim_matrix).argsort()[: k]
            for doc_id in doc_ids:
                doc_corduid = self.doc_ids[sim_id2doc_id[doc_id]]
                ranked_docs.append([sim_matrix[doc_id], doc_corduid])
            return ranked_docs

    def fit(self, n_iter, n):
        self.svd = TruncatedSVD(n_components=1000, random_state=0, n_iter=50)
        print("Fitting lda...")
        self.svd.fit(self.tf_idf)
        self.doc_mat = self.svd.transform(self.tf_idf)


"""
JOBLIB_TEMP_FOLDER=/tmp python evaluate.py --alg lda --operation calculate  --filename "./assets/svd_100_1" ; JOBLIB_TEMP_FOLDER=/tmp python evaluate.py --alg lda --operation eval  --filename "./assets/svd_100_1" ; bash eval.sh "run.txt"
"""