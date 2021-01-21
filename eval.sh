#/bin/sh

# assuming you have the following folder strcuture:
# in the parent directory of this directory (trec_covid):
#   you have folder trec_eval with executable trec_eval: 
#        ../trec_eval/trec_eval
#   you have perl script check_sub.pl
#       ../check_sub.pl
#   you have qrels file: qrels-covid_d5_j0.5-5.txt
#       ../qrels-covid_d5_j0.5-5.txt
fname=$1
../trec_eval/trec_eval ../qrels-covid_d5_j0.5-5.txt results/$fname -m all_trec > scores/$fname
cat scores/$fname # | egrep  '(success_10 |ndcg |map| P_10)'
