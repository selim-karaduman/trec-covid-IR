# trec_covid
Trec Covid Info Retrieval





## Fitting:
```bash
# Keep the file names as they are, they are used in the code as well
python evaluate.py --alg bm25 --operation calculate  --filename "./assets/bm25"
python evaluate.py --alg tfidf --operation calculate  --filename "./assets/tfidf"
python evaluate.py --alg svd --operation calculate  --filename "./assets/svd"
python evaluate.py --alg bert --operation calculate  --filename "./assets/bert" --bert-base-alg bm25
```

## Testing:
```bash
python evaluate.py --alg bert --operation eval  --filename "./assets/bert" --bert-base-alg tfidf --k -1
python evaluate.py --alg bert --operation eval  --filename "./assets/bert" --bert-base-alg bm25 --k -1
python evaluate.py --alg bm25 --operation eval  --filename "./assets/bm25"  --k 10
python evaluate.py --alg tfidf --operation eval  --filename "./assets/tfidf"  --k 10
python evaluate.py --alg svd --operation eval  --filename "./assets/svd_1000"  --k -1
```


