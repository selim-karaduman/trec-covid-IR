import xml.etree.ElementTree as ET
from base import CosineSimilarityBaseline
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

def eval_topics(topics, trec_ir):
    evals = []
    for step, (_, t) in enumerate(topics.items()):
        q = " ".join([t.query, t.question, t.narrative])
        eval_tuples = trec_ir.get_ranked_docs(q)
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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', choices=['random', 'cosine'], 
                            required=True)
    parser.add_argument('--operation', choices=['eval', 'calculate'], 
                            required=True)
    parser.add_argument('--filename', default="assets/cos_full")
    args = parser.parse_args()
    
    if args.alg == "cosine":
        trec_ir = CosineSimilarityBaseline()
        if args.operation == "eval":
            topics = load_topics(retrieve="even")
            trec_ir.load(args.filename)
            evals = eval_topics(topics, trec_ir)
            output = ("\n".join(evals))
            with open("results/run.txt", "w") as f:
                f.write(output)
        elif args.operation == "calculate":
            import pandas as pd
            data = pd.read_csv("metadata.csv")
            trec_ir.extract_stats_to_file(data, args.filename)
    else:
        raise ValueError
    
