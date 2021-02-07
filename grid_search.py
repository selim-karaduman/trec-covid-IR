from evaluate import *
from bert import *

def grid_search_bert(trec_ir, fname):
    # grid search for a parameter in range 0-1
    k = -1
    depth = 3
    N = 10
    delta = 1
    best_score = -1
    best_a = -1
    alpha = 0
    topics = load_topics(retrieve="even")
    trec_ir.load(fname)
    sim_matrices = dict()
    queries = dict()
    for d in range(depth):
        print("Depth: {}".format(d))
        delta /= N
        if d != 0:
            alpha = best_a - delta * N + delta
        N_ = N+1 if d == 0 else 2*N-1
        for _ in range(N_):
            evals = []
            for step, (_, t) in enumerate(topics.items()):
                if t.number not in queries:
                    query = " ".join([t.query, t.question, t.narrative])
                    queries[t.number] = query
                query = queries[t.number]
                if query not in sim_matrices:
                    bert_sim_matrix, sim_id2doc_id = \
                            trec_ir.get_sim_matrix(query, k)
                    base_sim_matrix, _ = trec_ir.base.get_sim_matrix(query, k)
                    sim_matrices[query] = (bert_sim_matrix, base_sim_matrix)
                bert_sim_matrix, base_sim_matrix = sim_matrices[query]
                sim_matrix = (alpha * bert_sim_matrix 
                                + (1 - alpha) * base_sim_matrix)
                eval_tuples = trec_ir.base\
                                .get_sorted_docs(sim_matrix, sim_id2doc_id, k)
                for i, (score, doc) in enumerate(eval_tuples):
                    # eval_tuples is sorted already
                    rank = i+1
                    run_tag = "0"
                    # rank is per topic: TODO check that
                    line = "{} Q0 {} {} {} {}".format(t.number, doc, 
                                                        rank, score, run_tag)
                    evals.append(line)
                print("{} / {}".format(step, len(topics)))
            output = ("\n".join(evals))
            with open("results/run.txt", "w") as f:
                f.write(output)
            score = get_map_from_runfile()
            if score > best_score:
                print("New best score: {}; alpha: {}".format(score, alpha))
                best_score = score
                best_a = alpha
            print("Candidate: {} ; Score: {}".format(alpha, score))
            alpha += delta
    return best_a


if __name__ == '__main__':
    ## base = TfIdfBaseline()
    ## base.load("assets/tfidf")
    base = BM25Baseline()
    base.load("assets/bm25")
    trec_ir = BertRanker(base)
    best_x = grid_search_bert(trec_ir, "assets/bertnotclean")
    print("Best found parameter is: {}".format(best_x))

"""
For TfIdf baseline:
Best found parameter is: 0.21 with MAP: 0.2959

"""