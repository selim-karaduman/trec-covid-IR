import numpy as np
import random
import scipy
from sortedcontainers import SortedList
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from document import Document
import pickle



class CosineSimilarityBaseline:
    def __init__(self, corpus=None, process_type="stem", 
                    fname="cosine_similarity", load=False):
        if load:
            self.load(fname)
            return None
        self.corpus = corpus
        self.process_type = process_type
        if process_type == "stem":
            self.porter = PorterStemmer()
            self.proc_token = lambda x: self.porter.stem(x)
        elif process_type == "lemmatize":
            self.lemmatizer = WordNetLemmatizer()
            self.proc_token = lambda x: self.lemmatizer.lemmatize(x)
        else:
            raise ValueError
        self.process_corpus_()
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
        self.porter = PorterStemmer()
        self.proc_token = lambda x: self.porter.stem(x)
  
    def get_ranked_docs(self, query):
        # todo: stopword elim to speed up
        tokenized_text = self.tokenize_(query)
        tokenized_text = [w for w in tokenized_text 
                            if not w in stopwords.words()]
        tokenized_text = self.process_(tokenized_text)
        query_doc = Document(tokenized_text, self.word2id)
        # query_doc = self.process_text_(query)
        
        # idf: shape: (V, 1)
        # doc_tf: (1, V)
        # q_tf_idf: (V, 1)
        q_tf_idf = self.idf.multiply(query_doc.tf_vec.T)
        ranked_docs = SortedList(key=lambda x: -x[0])
        doc_ids = set.union(*[self.posting_list.get(word, set())
                                  for word in query_doc.tokens])
        """
        for i, doc_id in enumerate(doc_ids):
            doc_cordid = self.doc_ids[doc_id]
            doc_tf_idf = self.tf_idf[doc_id]
            sim = cosine_similarity(q_tf_idf.T, doc_tf_idf)
            ranked_docs.add((sim[0, 0], doc_cordid))
            if i % 100 == 0:
                print("{}/{}; {} %".format(i, len(doc_ids), 
                                            i/len(doc_ids)*100
                                            ), flush=True)
        """
        sim_matrix = cosine_similarity(q_tf_idf.T, self.tf_idf)
        # (1, N)
        #doc_ids = np.argpartition(sim_matrix, -1000)[0, -1000:]
        doc_ids = (-sim_matrix).argsort()[0, :2000]
        doc_set = set()
        ranked_docs = []
        for doc_id in doc_ids:
            doc_cordid = self.doc_ids[doc_id]
            if len(doc_set) == 1000:
                break
            if doc_cordid not in doc_set:
                ranked_docs.append((sim_matrix[0, doc_id], doc_cordid))
                doc_set.add(doc_cordid)
        
        return ranked_docs
    
    def tokenize_(self, text):
        return [word for word in word_tokenize(text) 
                      if word.isalnum()]
    
    def process_(self, tokens):
        return [self.porter.stem(t) for t in tokens]
        # return [self.proc_token(t) for t in tokens]
        
    def process_text_(self, text):
        tokenized_text = self.tokenize_(text)
        tokenized_text = self.process_(tokenized_text)
        return Document(tokenized_text, self.word2id)
        
    def process_corpus_(self):
        self.docs = []
        self.doc_ids = []
        N = len(self.corpus)
        # 1 for UNK
        self.idf = dict({0: 1})
        self.word2id = dict()
        self.id2word = dict()
        self.cord2id = dict()
        self.id2cord = dict()
        self.posting_list = dict()
        # posting_list: key: string, value: set
        word_index = 1
        doc_index = 0
        for i in range(N):
            cord_uid = self.corpus["cord_uid"][i]
            title = self.corpus["title"][i]
            abstract = self.corpus["abstract"][i]
            if type(title).__name__=="float":
                title = ""
            if type(abstract).__name__ == "float":
                abstract = ""
            text= title + " " + abstract
            tokenized_text = self.tokenize_(text)
            tokenized_text = self.process_(tokenized_text)
            for word in tokenized_text:
                # add word to dictionary
                if word not in self.word2id:
                    self.word2id[word] = word_index
                    self.id2word[word_index] = word
                    word_index += 1
                # add doc to posting_list
                if word not in self.posting_list:
                    self.posting_list[word] = set()
                self.posting_list[word].add(doc_index)
                
            if cord_uid not in self.cord2id:
                self.cord2id[cord_uid] = doc_index
                self.id2cord[doc_index] = cord_uid
                doc_index += 1
                
            doc = Document(tokenized_text, word2id=None, 
                           doc_id=self.cord2id[cord_uid], 
                           cord_uid=cord_uid, title=title, 
                           abstract=abstract)
            self.docs.append(doc)
            self.doc_ids.append(doc.cord_uid)
            for word, tf in doc.tf_dict.items():
                index = self.word2id[word]
                if index not in self.idf: 
                    self.idf[index] = 0
                self.idf[index] += 1

            if i % 100 == 0:
                print("{} / {}; {} %".format(i, N, i/N*100))
        
        print("Processed files. Now getting tf-idf statistics")
        # idf: (V, 1);
        for i, doc in enumerate(self.docs):
            doc.tf_vec = doc.get_tf_vector_(self.word2id)
            if i % 100 == 0:
                print("{} / {}; {} %".format(i, N, i/N*100))
        self.idf = np.array(list(zip(*sorted(self.idf.items())))[1])
        self.idf = scipy.sparse.csr_matrix(np.log10(N/self.idf)).T
        # tf: (N, V)
        self.doc_tf = scipy\
                        .sparse.vstack([doc.tf_vec for doc in self.docs])
        # tf_idf: (N, V)
        self.tf_idf = (self.idf.multiply(self.doc_tf.T)).T



class RandomBaseline:
    def __init__(self, corpus):
        self.docs = []
        for i in range(len(corpus)):
            cord_uid = corpus["cord_uid"][i]
            doc = Document([], word2id=None, 
                           doc_id=0, 
                           cord_uid=cord_uid)
            self.docs.append(doc)

    def get_ranked_docs(self, query):
        docs = random.sample(self.docs, 1000)
        return [(i, doc.cord_uid) for i, doc in enumerate(docs)]
