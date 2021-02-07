import xml.etree.ElementTree as ET
from tfidf import TfIdfBaseline
from svd import SvdBaseline
from bert import BertRanker
from bm25 import BM25Baseline

class Topic:
    def __init__(self, number, query, question, narrative):
        self.number = number
        self.query = query
        self.question = question
        self.narrative = narrative
    
    def __repr__(self):
        return ('<topic number="{}">\n'
                '<query>{}</query>\n'
                '<question>{}</question>\n'
                '<narrative>{}</narrative>\n'
            '</topic>'
            ).format(self.number, self.query, self.question, self.narrative)


def load_topics(retrieve="all"):
    tree = ET.parse("topics-rnd5.xml")
    root = tree.getroot()
    topics = dict()
    for topic in root:
        t_id = topic.get("number")
        if (retrieve == "even") and (int(t_id) % 2 != 0):
            continue
        elif (retrieve == "odd") and (int(t_id) % 2 == 0):
                continue

        kwargs = dict()
        kwargs["number"] = t_id
        for c in topic:
            element_type = c.tag
            element_content = c.text
            element_content = element_content.strip()
            kwargs[element_type] = element_content
        topics[t_id] = Topic(**kwargs)
    return topics

def eval_topics(topics, trec_ir, k):
    evals = []
    for step, (_, t) in enumerate(topics.items()):
        q = " ".join([t.query, t.question, t.narrative])
        eval_tuples = trec_ir.get_ranked_docs(q, k=k)
        for i, (score, doc) in enumerate(eval_tuples):
            # eval_tuples is sorted already
            rank = i+1
            run_tag = "0"
            # rank is per topic: TODO check that
            line = "{} Q0 {} {} {} {}".format(t.number, doc, 
                                                rank, score, run_tag)
            evals.append(line)
        print("{} / {}".format(step, len(topics)))
    return evals

def get_queries_from_topics(topics):
    queries = []
    for _, t in topics.items():
        q = " ".join([t.query, t.question, t.narrative])
        queries.append(q)
    return queries

def gen_runfile(trec_ir, fname, k):
    topics = load_topics(retrieve="even")
    trec_ir.load(fname)
    evals = eval_topics(topics, trec_ir, k)
    output = ("\n".join(evals))
    with open("results/run.txt", "w") as f:
        f.write(output)

def get_map_from_runfile():
    # runs trec_eval process
    import subprocess
    a = subprocess.run(["bash", "eval.sh", "run.txt"], stdout=subprocess.PIPE)
    data = a.stdout.decode("utf-8").replace(" ", "")
    res = {l.split("\t")[0]: float(l.split("\t")[2]) 
                for l in data.split("\n") if l}
    map_score = res["map"]
    return map_score

def get_relevant_scores():
    import subprocess
    a = subprocess.run(["bash", "eval.sh", "run.txt"], stdout=subprocess.PIPE)
    data = a.stdout.decode("utf-8").replace(" ", "")
    res = {l.split("\t")[0]: float(l.split("\t")[2]) 
                for l in data.split("\n") if l}
    map_score = res["map"]
    p10 = res["P_10"]
    ndcg = res["ndcg"]
    success_10 = res["success_10"]
    print("MAP: {}\nP_10: {}\nNDCG: {}\nsuccess_10: {}"\
            .format(map_score, p10, ndcg, success_10))
    return (map_score, p10, ndcg, success_10)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', choices=['random', 'tfidf', 
                                            'svd', 'bert', 'bm25'], 
                            required=True)
    parser.add_argument('--operation', choices=['eval', 'calculate'], 
                            required=True)
    parser.add_argument('--filename', required=True)
    parser.add_argument('--k', default=-1, type=int)
    parser.add_argument('--bert-base-alg')
    args = parser.parse_args()
    
    if args.alg == "tfidf":
        trec_ir = TfIdfBaseline()
        if args.operation == "eval":
            gen_runfile(trec_ir, args.filename, args.k)
            get_relevant_scores()
        elif args.operation == "calculate":
            import pandas as pd
            data = pd.read_csv("metadata.csv")
            trec_ir.extract_stats_to_file(data, args.filename)
    elif args.alg == "svd":
        trec_ir = SvdBaseline("assets/tfidf")
        if args.operation == "eval":
            gen_runfile(trec_ir, args.filename, args.k)
            get_relevant_scores()
        elif args.operation == "calculate":
            trec_ir.extract_stats_to_file(args.filename)
    elif args.alg == "bert":
        if args.bert_base_alg == "tfidf":    
            base = TfIdfBaseline()
            base.load("assets/tfidf")
        elif args.bert_base_alg == "bm25":
            base = BM25Baseline()
            base.load("assets/bm25")
        else:
            print("With bert use one of < tfidf, bm25 > with --bert-base-alg")
            raise ValueError
        trec_ir = BertRanker(base)
        if args.operation == "eval":
            gen_runfile(trec_ir, args.filename, args.k)
            get_relevant_scores()
        elif args.operation == "calculate":
            import pandas as pd
            data = pd.read_csv("metadata.csv")
            topics = load_topics()
            queries = get_queries_from_topics(topics)
            trec_ir.extract_stats_to_file(data, queries, args.filename)
    elif args.alg == "bm25":
        trec_ir = BM25Baseline(b=0.75, k=1.5)
        if args.operation == "eval":
            gen_runfile(trec_ir, args.filename, args.k)
            get_relevant_scores()
        elif args.operation == "calculate":
            import pandas as pd
            data = pd.read_csv("metadata.csv")
            trec_ir.extract_stats_to_file(data, args.filename)    
    else:
        raise ValueError
    
