import pickle
import numpy as np
from config import INDEX_DIR


def tokenize_corpus(corpus, col="text_clean"):
    return [doc.split() for doc in corpus[col].fillna("")]


def mean_vector(tokens, model, dim=300):
    vecs = [model[t] for t in tokens if t in model]
    return np.mean(vecs, axis=0) if vecs else np.zeros(dim)


def build_doc_matrix(tokenized, model, dim=300):
    return np.vstack([mean_vector(tokens, model, dim) for tokens in tokenized])


def save_index(obj, name):
    path = INDEX_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  сохранён: {path}")


def load_index(name):
    path = INDEX_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)
