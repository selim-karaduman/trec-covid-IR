import numpy as np
import re
import random
import scipy
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from document import Document
import pickle
from abc import ABC, abstractmethod

PUNCS = "!\"#$%&()*+,-./:;<=>?@[\\]^_{|}~"
PUNCS_RE = re.compile("([{}])".format(PUNCS), re.IGNORECASE | re.DOTALL)
STOPWORDS_SET = set(stopwords.words('english'))
THRESHOLD_MIN_TOKEN = 1

class Base(ABC):
    def __init__(self):
        self.porter = PorterStemmer()

    @abstractmethod
    def extract_stats_to_file(self, *args, **kwargs):
        pass

    @abstractmethod
    def save(self, *args, **kwargs):
        pass

    @abstractmethod
    def load(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_ranked_docs(self, *args, **kwargs):
        pass

    def calculate_sim_vector(self, matrix, q_v, k, tokenized_text):
        if k == -1:
            sim_vector = cosine_similarity(matrix, q_v)[:, 0]
            return sim_vector, None

        cand_doc_ids = np.array(list
                            (set.union(
                                *[self.posting_list.get(word, set())
                                    for word in tokenized_text])
                            )
                        )
        sim_id2doc_id = {i:d_id for i, d_id in enumerate(cand_doc_ids)}
        sim_vector = cosine_similarity(matrix[cand_doc_ids, :], 
                                        q_v)[:, 0]
        return sim_vector, sim_id2doc_id

    def get_sorted_docs(self, sim_vector, sim_id2doc_id, k):
        # sim_vector is 1d np array
        doc_ids = (-sim_vector).argsort()
        if k != -1:
            doc_ids = doc_ids[: k]
        ranked_docs = []
        if sim_id2doc_id is None:
            for doc_id in doc_ids:
                doc_corduid = self.doc_ids[doc_id]
                ranked_docs.append([sim_vector[doc_id], doc_corduid])
            return ranked_docs
        else:
            for doc_id in doc_ids:
                doc_corduid = self.doc_ids[sim_id2doc_id[doc_id]]
                ranked_docs.append([sim_vector[doc_id], doc_corduid])
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
        self.d_avg = 0
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
            self.d_avg += len(tokenized_text)
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
        self.d_avg /= len(self.doc_ids)
        return idf, docs


class RandomBaseline:
    def __init__(self, corpus):
        self.docs_ids = []
        for i in range(len(corpus)):
            cord_uid = corpus["cord_uid"][i]
            self.docs_ids.append(cord_uid)

    def get_ranked_docs(self, query, k=-1):
        if k == -1:
            docs = random.sample(self.docs_ids, len(self.doc_ids))
        else:
            docs = random.sample(self.docs_ids, k)
        return [(i, cord_uid) for i, cord_uid in enumerate(docs)]


