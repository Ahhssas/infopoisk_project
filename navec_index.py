import numpy as np
from navec import Navec
from utils import tokenize_corpus, build_doc_matrix, save_index
from config import NAVEC_PATH, W2V_DIM


def build_navec(corpus):
    """
    Предобученные русские эмбеддинги Navec
    Модель: navec_hudlit_v1 — худлит, 12B токенов, 300d
    Источник: https://github.com/natasha/navec (Natasha Project)
    """
    navec = Navec.load(NAVEC_PATH)
    tokenized = tokenize_corpus(corpus)
    doc_vectors = build_doc_matrix(tokenized, navec, dim=W2V_DIM)
    save_index({"doc_vectors": doc_vectors}, "navec_index")
    print(f"Navec: матрица {doc_vectors.shape}")
    return doc_vectors
