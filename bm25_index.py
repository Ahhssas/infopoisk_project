import pandas as pd
from rank_bm25 import BM25Okapi
from utils import tokenize_corpus, save_index


def build_bm25(corpus):
    """
    BM25Okapi индекс.
    Библиотека: rank_bm25 (https://github.com/dorianbrown/rank_bm25)
    """
    tokenized = tokenize_corpus(corpus)
    index = BM25Okapi(tokenized)
    save_index({"bm25": index, "tokenized": tokenized}, "bm25_index")
    print(f"BM25: {len(tokenized):,} документов проиндексировано.")
    return index
