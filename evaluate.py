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


def load_topics():
    tree = ET.parse("topics-rnd5.xml")
    root = tree.getroot()
    topics = dict()
    for topic in root:
        t_id = topic.get("number")
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
            run_tag = "GROUP_NAME"
            # rank is per topic: TODO check that
            line = "{} Q0 {} {} {} {}".format(t.number, doc, 
                                                rank, score, run_tag)
            evals.append(line)
        print("{} / {}".format(step, len(topics)))
    return evals

if __name__ == '__main__':
    trec_ir = CosineSimilarityBaseline(fname="assets/cosine_similarity", 
                                        load=True)
    topics = load_topics()
    lines  = eval_topics(topics, trec_ir)
    output = ("\n".join(lines))
    with open("results/run.txt", "w") as f:
        f.write(output)
    
    

