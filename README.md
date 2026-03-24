# Проект по информационному поиску

## Структура проекта
```
ir_project/
├── config.py          # пути и гиперпараметры
├── preprocessing.py   # очистка и лемматизация текста
├── utils.py           # общие функции (токенизация, векторизация, сохранение)
├── bm25_index.py      # построение BM25-индекса
├── w2v_index.py       # обучение Word2Vec и построение индекса
├── navec_index.py     # построение индекса на основе Navec
├── build_indexes.py   # точка входа — строит все три индекса
├── search.py          # поисковые функции для каждого метода
├── cli.py             # интерфейс командной строки
└── indexes/           # сохранённые индексы (*.pkl)
```

---

## Модели

| Индекс   | Библиотека   | Описание |
|----------|--------------|----------|
| BM25     | `rank_bm25`  | BM25Okapi|
| Word2Vec | `gensim`     | Skip-gram|
| Navec    | `navec`      | Предобученные русские эмбеддинги| 

Для BM25 релевантность считается через встроенную скоринговую функцию `get_scores()`.
Для Word2Vec и Navec — косинусное сходство между вектором запроса и матрицей документов.

---

## Установка
```bash
pip install pymorphy3 razdel nltk rank-bm25 gensim navec

# скачать модель Navec
wget https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar
```
---

## Как запустить

### 1. Предобработка и построение индексов
```python
import pandas as pd
from preprocessing import preprocess_corpus
from build_indexes import build_all

df = pd.read_csv("jokes_2000.csv")
corpus_clean = preprocess_corpus(df, text_col="text")
indexes = build_all(corpus_clean)
```

### 2. Поиск через Python
```python
from search import search

results = search("муж жена дома", corpus_clean, method="bm25", top_k=5)
```

### 3. Поиск через CLI
```bash
python cli.py --query "муж жена дома" --method bm25 --top_k 5
```

### Примеры
```bash
# базовый поиск
python cli.py -q "врач пациент больница"

# топ-10 через Word2Vec
python cli.py -q "школа учитель урок" -m w2v -k 10

# поиск через Navec с другим корпусом
python cli.py -q "кот собака" -m navec -c /content/my_corpus.csv
```

---

## Вывод результатов

При каждом поиске выводится время выполнения и список документов с оценкой релевантности.

Диапазон оценок:
- BM25 — от 0 и выше, чем больше тем релевантнее (несопоставимо между запросами)
- Word2Vec / Navec — косинусное сходство от 0 до 1, где 1 — полное совпадение
