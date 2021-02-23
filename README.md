# Trec-Covid task: Information Retrieval System

### Algorithms:
* Tf-Idf based cosine similarity scoring
* BM25 scoring
* SVD based cosine similarity scoring
* Re-ranking with pre-trained bert model
    * [gsarti/covidbert-nli](https://huggingface.co/gsarti/covidbert-nli)

* Notes:
    * TfIdf and BM25 are implemented using scipy
    * SVD is implemented using sklearn's TruncatedSVD
    * Re-ranking is implemented using sentence-transformers library

***

### Requirements:
* pandas >= 0.25.1
* scipy >= 1.4.1
* numpy >= 1.17.2
* scikit-learn >= 0.21.3 
* sentence-transformers >= 0.4.1.2 
* nltk >= 3.5
* joblib >= 0.13.2
* and the requirements for these packages

***

### Setup:
```bash
mkdir assets models results scores
## wget "https://ir.nist.gov/covidSubmit/data/topics-rnd5.xml"
wget "https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_2020-07-16.tar.gz"
tar -xvf cord-19_2020-07-16.tar.gz "2020-07-16/metadata.csv"
mv "2020-07-16/metadata.csv" .
rm -rf cord-19_2020-07-16.tar.gz 2020-07-16/
# install relevance scores
## wget "https://ir.nist.gov/covidSubmit/data/qrels-covid_d5_j0.5-5.txt"

# install trec_eval
git clone https://github.com/usnistgov/trec_eval.git
cd trec_eval
make
cd ..
```

***

### Fitting:
```bash
# Keep the file names as they are, they are used in the code as well
# Order is important since svd uses tfidf and bert uses tfidf and bm25 as its baseline
# By default SVD has 100 dimensions with 20 iterations which works bad, but otherwise training take too long
# Takes a LONG (especially bert part) time
python evaluate.py --alg tfidf --operation calculate  --filename "./assets/tfidf"
python evaluate.py --alg bm25 --operation calculate  --filename "./assets/bm25"
python evaluate.py --alg svd --operation calculate  --filename "./assets/svd"
# base model doesn't matter
python evaluate.py --alg bert --operation calculate  --filename "./assets/bert" --bert-base-alg bm25
```

***


### Testing:
```bash
python evaluate.py --alg bert --operation eval  --filename "./assets/bert" --bert-base-alg tfidf --k 1000
python evaluate.py --alg bert --operation eval  --filename "./assets/bert" --bert-base-alg bm25 --k 1000
python evaluate.py --alg bm25 --operation eval  --filename "./assets/bm25"  --k 1000
python evaluate.py --alg tfidf --operation eval  --filename "./assets/tfidf"  --k 1000
python evaluate.py --alg svd --operation eval  --filename "./assets/svd"  --k 1000
```

***


### Results:
|                   | **NDCG**      | **P@10**      | **bpref**     | **map**       |
|:-------------     |:------------- |:------------- |:------------- |:------------- |
| BM25              | 0.5165        | 0.786         | 0.406         | 0.2861        |
| TD-IDF            | 0.4477        | 0.578         | 0.3813        | 0.207         |
| SVD               | 0.2529        | 0.308         | 0.2391        | 0.0873        |
| BERT ON TF-IDF    | 0.4601        | 0.602         | 0.3912        | 0.2193        |
| **BERT ON BM25**  | 0.5228        | 0.796         | 0.4121        | 0.291         |

