import subprocess
from evaluate import *
from tfidf import TfIdfBaseline
from svd import SvdBaseline
from bert import BertRanker
from bm25 import BM25Baseline
import numpy as np
from scipy import stats

def paired_test(trec_ir1, file1, retr1, name1, trec_ir2, file2, retr2, name2):
    gen_runfile(trec_ir1, file1, -1, retr1)
    a = subprocess.run(["../trec_eval/trec_eval",
                            "-q", "../qrels-covid_d5_j0.5-5.txt", 
                            "results/run.txt", "-m", "all_trec"
                            ], stdout=subprocess.PIPE)
    data = a.stdout.decode("utf-8").replace(" ", "")
    res1 = {}
    maps1 = []
    for l in data.split("\n"):
        if not l: 
            continue
        a, qid, b = l.split("\t")
        if not qid.isnumeric():
            continue
        if a != "map":
            continue
        res1[int(qid)] = float(b)

    for k in sorted(res1):
        maps1.append(res1[k])

    gen_runfile(trec_ir2, file2, -1, retr2)
    a = subprocess.run(["../trec_eval/trec_eval",
                            "-q", "../qrels-covid_d5_j0.5-5.txt", 
                            "results/run.txt", "-m", "all_trec"
                            ], stdout=subprocess.PIPE)
    data = a.stdout.decode("utf-8").replace(" ", "")
    res2 = {}
    maps2 = []
    for l in data.split("\n"):
        if not l: 
            continue
        a, qid, b = l.split("\t")
        if not qid.isnumeric():
            continue
        if a != "map":
            continue
        res2[int(qid)] = float(b)

    for k in sorted(res2):
        maps2.append(res2[k])

    maps1 = np.array(maps1)
    maps2 = np.array(maps2)
    print("Evaluation is done. Performing test...")
    t, p = stats.ttest_rel(maps1, maps2)
    print("t-statistics: {} ; p-value: {}".format(t, p))
    if p > 0.1:
        print("<{}> and <{}> have no significant difference".format(name1,
                                                                     name2))
    else:
        m1 = maps1.mean()
        m2 = maps2.mean()
        if m1 > m2:
            print("<{}> is significantly better than <{}>".format(name1, 
                                                                    name2))
        else:
            print("<{}> is significantly better than <{}>".format(name2, 
                                                                    name1))


base1 = BM25Baseline()
base1.load("assets/bm25")
trec_ir1 = BertRanker(base1)

print("Paired test between tfidf")
trec_ir2 = TfIdfBaseline()
paired_test(trec_ir1, "assets/bert", "even", "bert on bm25",
                trec_ir2, "assets/tfidf", "even", "tfidf")
print("*"*100)
print("*"*100)
# <even set> and <odd set> have no significant difference
print("Paired test between bm25")
trec_ir2 = BM25Baseline(b=0.75, k=1.5)
paired_test(trec_ir1, "assets/bert", "even", "bert on bm25", 
                trec_ir2, "assets/bm25", "even", "bm25")
print("*"*100)
print("*"*100)

print("Paired test between svd")
trec_ir2 = SvdBaseline("assets/tfidf")
paired_test(trec_ir1, "assets/bert", "even", "bert on bm25",
                trec_ir2, "assets/svd_1000", "even", "svd")
print("*"*100)
print("*"*100)


print("Paired test between bert on tfidf")
base2 = TfIdfBaseline()
base2.load("assets/tfidf")
trec_ir2 = BertRanker(base2)
paired_test(trec_ir1, "assets/bert", "even", "bert on bm25", 
                trec_ir2, "assets/bert", "even", "bert on tfidf")
print("*"*100)
print("*"*100)
