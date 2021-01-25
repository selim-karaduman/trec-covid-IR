import scipy
import scipy.sparse
import numpy as np
from collections import Counter

class Document:
    def __init__(self, tokens):
        self.tf_dict = Counter(tokens)
            
    def cache_tf_vector(self, word2id):
        V = len(word2id)
        tf_vec = np.zeros((V))
        for tok, count in self.tf_dict.items():
            if tok in word2id:
                tf_vec[word2id[tok]] = count
        tf_vec[tf_vec>0] = np.log(tf_vec[tf_vec>0]) + 1
        tf_vec = scipy.sparse.csr_matrix(tf_vec)
        self.tf_vec = tf_vec
        return tf_vec
        