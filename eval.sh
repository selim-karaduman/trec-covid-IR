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
echo $fname

# python evaluate.py

# Check the output of run.txt.errlog manually
perl ../check_sub.pl results/$fname

../trec_eval/trec_eval -c -M1000 -m all_trec ../qrels-covid_d5_j0.5-5.txt results/$fname > scores/$fname
# | egrep '(ndcg_cut_10 |recall_1000 )'



cat scores/$fname # | egrep  '(success_10 |ndcg |map| P_10)'
































