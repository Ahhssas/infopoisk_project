from pathlib import Path

INDEX_DIR = Path("/content/indexes")
INDEX_DIR.mkdir(exist_ok=True)

NAVEC_PATH = "/content/navec_hudlit_v1_12B_500K_300d_100q.tar"
W2V_DIM = 300
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
W2V_EPOCHS = 10
W2V_SEED = 42
