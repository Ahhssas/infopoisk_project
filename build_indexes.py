from bm25_index import build_bm25
from w2v_index import build_word2vec
from navec_index import build_navec


def build_all(corpus):
    print("Построение индексов\n")

    print("[1/3] BM25...")
    bm25 = build_bm25(corpus)

    print("\n[2/3] Word2Vec...")
    w2v_vectors = build_word2vec(corpus)

    print("\n[3/3] Navec...")
    navec_vectors = build_navec(corpus)

    return {
        "bm25": bm25,
        "w2v_vectors": w2v_vectors,
        "navec_vectors": navec_vectors,
    }
