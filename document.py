import scipy
import scipy.sparse
import numpy as np
from collections import Counter

class Document:
    def __init__(self, tokens, word2id=None, doc_id=None, 
                 cord_uid=None, abstract=None, title=None):
        self.tokens = tokens
        self.doc_id = doc_id
        self.cord_uid = cord_uid
        self.abstract = abstract
        self.title = title
        self.tf_dict = Counter(tokens)
        if word2id:
            self.tf_vec = self.get_tf_vector_(word2id)
            
    def get_tf_vector_(self, word2id):
        tokens = self.tokens
        V = len(word2id) + 1
        tf_vec = [0]*V
        for token in tokens:
            if token in word2id:
                tf_vec[word2id[token]] += 1
        tf_vec = np.array(tf_vec)
        tf_vec[tf_vec>0] = np.log10(tf_vec[tf_vec>0]) + 1
        # store as sparse matrix
        tf_vec = scipy.sparse.csr_matrix(tf_vec)
        return tf_vec
