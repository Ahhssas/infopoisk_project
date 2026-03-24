import time
import numpy as np
from utils import load_index, mean_vector
from preprocessing import preprocess_text
from config import W2V_DIM


def cosine_similarity(query_vec, doc_matrix):
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    doc_norms = np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-10
    return doc_matrix.dot(query_norm) / doc_norms.squeeze()


def search_bm25(query, corpus, top_k=10):
    start = time.time()
    data = load_index("bm25_index")
    bm25 = data["bm25"]
    tokens = preprocess_text(query).split()
    scores = bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:top_k]
    results = corpus.iloc[top_idx].copy()
    results["score"] = scores[top_idx]
    elapsed = time.time() - start
    print(f"  время поиска: {elapsed:.4f} сек.")
    return results[["text", "score"]].reset_index(drop=True)


def search_w2v(query, corpus, top_k=10):
    start = time.time()
    data = load_index("w2v_index")
    doc_vectors = data["doc_vectors"]
    model = data["model"]
    tokens = preprocess_text(query).split()
    query_vec = mean_vector(tokens, model.wv, W2V_DIM)
    scores = cosine_similarity(query_vec, doc_vectors)
    top_idx = np.argsort(scores)[::-1][:top_k]
    results = corpus.iloc[top_idx].copy()
    results["score"] = scores[top_idx]
    elapsed = time.time() - start
    print(f"  время поиска: {elapsed:.4f} сек.")
    return results[["text", "score"]].reset_index(drop=True)


def search_navec(query, corpus, top_k=10):
    from navec import Navec
    from config import NAVEC_PATH
    start = time.time()
    data = load_index("navec_index")
    doc_vectors = data["doc_vectors"]
    navec = Navec.load(NAVEC_PATH)
    tokens = preprocess_text(query).split()
    query_vec = mean_vector(tokens, navec, W2V_DIM)
    scores = cosine_similarity(query_vec, doc_vectors)
    top_idx = np.argsort(scores)[::-1][:top_k]
    results = corpus.iloc[top_idx].copy()
    results["score"] = scores[top_idx]
    elapsed = time.time() - start
    print(f"  время поиска: {elapsed:.4f} сек.")
    return results[["text", "score"]].reset_index(drop=True)


def search(query, corpus, method="bm25", top_k=10):
    methods = {
        "bm25": search_bm25,
        "w2v": search_w2v,
        "navec": search_navec,
    }
    if method not in methods:
        raise ValueError(f"Неизвестный метод: {method}. Доступны: {list(methods.keys())}")
    return methods[method](query, corpus, top_k)
