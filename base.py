import numpy as np
import re
import random
import scipy
from sortedcontainers import SortedList
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from document import Document
import pickle

PUNCS = "!\"#$%&()*+,-./:;<=>?@[\\]^_{|}~"
PUNCS_RE = re.compile("([{}])".format(PUNCS), re.IGNORECASE | re.DOTALL)
STOPWORDS_SET = set(stopwords.words('english'))
THRESHOLD_MIN_TOKEN = 1

"""
To extract statistics etc. initially run:
    b = CosineSimilarityBaseline()
    b.extract_stats_to_file(corpus, outfname)
To load existing statistics:
    b = CosineSimilarityBaseline()
    b.load(fname)
"""
class CosineSimilarityBaseline:
    def __init__(self):
        self.porter = PorterStemmer()
        
    def extract_stats_to_file(self, corpus, fname):
        self.process_corpus(corpus)
        self.save(fname)

    def save(self, fname):
        t = (self.idf, self.posting_list, self.tf_idf, 
                self.doc_ids, self.word2id)
        with open(fname, "wb") as f:
            pickle.dump(t, f)

    def load(self, fname):
        with open(fname, "rb") as f:
            t = pickle.load(f)
        (self.idf, self.posting_list, self.tf_idf, 
            self.doc_ids, self.word2id) = t
        
  
    def get_ranked_docs(self, query):
        # this assumes that no mutual documents in the self.doc_ids
        tokenized_text = self.process_text(query)
        query_doc = Document(tokenized_text)
        query_doc.cache_tf_vector(self.word2id)
        
        # idf: shape: (V, V)
        # query_doc.tf_vec: (1, V)
        # q_tf_idf: (1, V)
        q_tf_idf = query_doc.tf_vec * self.idf
        cand_doc_ids = np.array(list
                                    (set.union(
                                        *[self.posting_list.get(word, set())
                                            for word in tokenized_text])
                                    )
                                )
        sim_matrix = cosine_similarity(self.tf_idf, q_tf_idf)[:, 0]
        doc_ids = (-sim_matrix).argsort()
        ranked_docs = []
        for doc_id in doc_ids:
            doc_corduid = self.doc_ids[doc_id]
            ranked_docs.append((sim_matrix[doc_id], doc_corduid))
        return ranked_docs
    
    def process_text(self, text):
        text = text.replace("-\n", " ")
        text = PUNCS_RE.sub(" ", text)
        text = text.lower()
        tokens = [word for word in word_tokenize(text) 
                      if (word.isalpha() and len(word) > 1)]
        tokens = [w for w in tokens if not w in STOPWORDS_SET]
        tokens = [self.porter.stem(t) for t in tokens]
        return tokens
        
    def process_corpus(self, corpus):
        docs = []
        idf = []
        cord_uids = set()
        self.doc_ids = []
        self.word2id = dict()
        self.posting_list = dict()
        # posting_list: key: string, value: set
        word_index = 0
        doc_index = 0
        for i in range(len(corpus)):
            cord_uid = corpus["cord_uid"][i]
            if cord_uid in cord_uids:
                continue
            title = corpus["title"][i]
            title = "" if (not isinstance(title, str)) else title
            abstract = corpus["abstract"][i]
            abstract = "" if (not isinstance(abstract, str)) else abstract
            text= title + " " + abstract
            tokenized_text = self.process_text(text)
            # if the document is very short, skip it
            if len(tokenized_text) < THRESHOLD_MIN_TOKEN:
                continue
            doc = Document(tokenized_text)
            for word in tokenized_text:
                # add word to dictionary
                if word not in self.word2id:
                    self.word2id[word] = word_index
                    idf.append(0)
                    word_index += 1
                # add doc to posting_list
                if word not in self.posting_list:
                    self.posting_list[word] = set()
                self.posting_list[word].add(doc_index)     
            docs.append(doc)
            self.doc_ids.append(cord_uid)
            cord_uids.add(cord_uid)
            doc_index += 1
            for word in doc.tf_dict.keys():
                index = self.word2id[word]
                idf[index] += 1
            if i % 100 == 0:
                print("{} / {}; {:.2f} %".format(i, len(corpus), 
                                                    i / len(corpus) * 100)) 
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


class RandomBaseline:
    def __init__(self, corpus):
        self.docs_ids = []
        for i in range(len(corpus)):
            cord_uid = corpus["cord_uid"][i]
            self.docs_ids.append(cord_uid)

    def get_ranked_docs(self, query):
        docs = random.sample(self.docs_ids, 1000)
        return [(i, cord_uid) for i, cord_uid in enumerate(docs)]

