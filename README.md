# trec_covid
Trec Covid Info Retrieval

### Algorithms:
* TfIdf based cosine similarity scoring
* BM25 scoring
* SVD based cosine similarity scoring
* Re-ranking with pre-trained bert model
    * [gsarti/covidbert-nli](https://huggingface.co/gsarti/covidbert-nli)

* Notes:
    * TfIdf and BM25 are implemented using scipy
    * SVD is implemented using sklearn's truncatedSVD
    * Re-ranking is implemented using sentence-transformers library

***
***

### Requirements:
* scipy >= 1.4.1
* numpy >= 1.17.2
* scikit-learn >= 0.21.3 
* sentence-transformers >= 0.4.1.2 
* and the requirements for these packages

***
***

### Setup:
```bash
mkdir assets models results scores
wget "https://ir.nist.gov/covidSubmit/data/topics-rnd5.xml"
wget "https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_2020-07-16.tar.gz"
tar -xvf cord-19_2020-07-16.tar.gz
mv "2020-07-16/metadata.csv" .
rm -rf cord-19_2020-07-16.tar.gz 2020-07-16/

# install trec_eval
pushd .
cd ..
git clone https://github.com/usnistgov/trec_eval.git
cd trec_eval
make
popd
```

***
***

### Fitting:
```bash
# Keep the file names as they are, they are used in the code as well
python evaluate.py --alg bm25 --operation calculate  --filename "./assets/bm25"
python evaluate.py --alg tfidf --operation calculate  --filename "./assets/tfidf"
python evaluate.py --alg svd --operation calculate  --filename "./assets/svd"
python evaluate.py --alg bert --operation calculate  --filename "./assets/bert" --bert-base-alg bm25
```

***
***


### Testing:
```bash
python evaluate.py --alg bert --operation eval  --filename "./assets/bert" --bert-base-alg tfidf --k -1
python evaluate.py --alg bert --operation eval  --filename "./assets/bert" --bert-base-alg bm25 --k -1
python evaluate.py --alg bm25 --operation eval  --filename "./assets/bm25"  --k -1
python evaluate.py --alg tfidf --operation eval  --filename "./assets/tfidf"  --k -1
python evaluate.py --alg svd --operation eval  --filename "./assets/svd_1000"  --k -1
```

***
***

### Results:
| ....          | BM25          | TFIDF         | SVD           | BERT ON TFIDF | BERT ON BM25  |
|:------------- |:------------- |:------------- |:------------- |:-------------:|:------------- |
| **MAP**       | 0.3485        | 0.2765        | 0.2219        | 0.2996        | 0.3547        |
| **P_10**      | 0.8           | 0.6080        | 0.304         | 0.636         | 0.8           |
| **NDCG**      | 0.7824        | 0.7462        | 0.7004        | 0.7587        | 0.7855        |


* The results are obtained for k==-1; which means: for each query program returns all documents sorted by their score. *Unlike* trec_coivd requirement ok k==1000.
