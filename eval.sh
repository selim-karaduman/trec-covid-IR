#/bin/sh

fname=$1
./trec_eval/trec_eval  ./qrels-covid_d5_j0.5-5.txt results/$fname -m all_trec > scores/$fname
cat scores/$fname # | egrep  '(success_10 |ndcg |map| P_10)'
