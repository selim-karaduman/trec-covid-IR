# trec_covid
Trec Covid Info Retrieval


To run cosine based:
```bash
python evaluate.py --alg cosine --operation calculate --filename "./assets/cos_full2"
python evaluate.py --alg cosine --operation eval --filename "assets/cos_full2" 
bash eval.sh "run_full_2.txt"
```