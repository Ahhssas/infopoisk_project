import numpy as np
from gensim.models import Word2Vec
from utils import tokenize_corpus, build_doc_matrix, save_index
from config import W2V_DIM, W2V_WINDOW, W2V_MIN_COUNT, W2V_EPOCHS, W2V_SEED


def build_word2vec(corpus):
    """
    Word2Vec Skip-gram, обучен на корпусе.
    Библиотека: gensim (https://radimrehurek.com/gensim/)
    """
    tokenized = tokenize_corpus(corpus)
    model = Word2Vec(
        sentences=tokenized,
        vector_size=W2V_DIM,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        workers=4,
        sg=1,
        epochs=W2V_EPOCHS,
        seed=W2V_SEED,
    )
    doc_vectors = build_doc_matrix(tokenized, model.wv, W2V_DIM)
    save_index({"model": model, "doc_vectors": doc_vectors}, "w2v_index")
    print(f"Word2Vec: словарь {len(model.wv):,} слов, матрица {doc_vectors.shape}.")
    return doc_vectors
