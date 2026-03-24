# Проект по информационному поиску
---

## Установка
```bash
pip install pymorphy3 razdel nltk rank-bm25 gensim navec

# скачать модель Navec
wget https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar
```
---

## Корпус

Использован датасет [Russian Jokes](https://www.kaggle.com/datasets/konstantinalbul/russian-jokes) с Kaggle.
```python
import kagglehub
path = kagglehub.dataset_download("konstantinalbul/russian-jokes")
```
Датасет содержит анекдоты на русском языке. После загрузки был применен пайплайн предобработки — приведение к нижнему регистру, удаление спецсимволов, замена ё→е, токенизация (razdel), фильтрация стоп-слов (NLTK + дополнительный словарь) и лемматизация (pymorphy3). Документы короче 3 слов и дубликаты были удалены.

Итоговый размер корпуса после предобработки: **124 762 документа**, сохранен в `jokes_clean.csv`.

## Структура проекта
```
├── config.py  # пути и гиперпараметры
├── preprocessing.py  # очистка и лемматизация текста
├── utils.py  # общие функции (токенизация, векторизация, сохранение)
├── bm25_index.py  # построение BM25-индекса
├── w2v_index.py  # обучение Word2Vec и построение индекса
├── navec_index.py  # построение индекса на основе Navec
├── build_indexes.py  # точка входа — строит все три индекса
├── search.py  # поисковые функции для каждого метода
├── cli.py  # интерфейс командной строки
└── indexes/  # сохраненные индексы (*.pkl)
```

## Модели

| Индекс   | Библиотека   | Описание |
|----------|--------------|----------|
| BM25     | `rank_bm25`  | BM25Okapi|
| Word2Vec | `gensim`     | Skip-gram|
| Navec    | `navec`      | Предобученные русские эмбеддинги| 

Для BM25 релевантность считается через встроенную скоринговую функцию `get_scores()`.

Для Word2Vec и Navec - косинусное сходство между вектором запроса и матрицей документов.

---

## Как запустить

### Шаг 1: предобработка корпуса
```python
import pandas as pd
from preprocessing import preprocess_corpus

df = pd.read_csv("jokes_2000.csv")
corpus_clean = preprocess_corpus(df, text_col="text")
corpus_clean.to_csv("jokes_clean.csv", index=False)
```

### Шаг 2 — построение индексов
```python
from build_indexes import build_all

indexes = build_all(corpus_clean)
```

В папке `indexes/` появятся три файла: `bm25_index.pkl`, `w2v_index.pkl`, `navec_index.pkl`. Их не нужно пересобирать при каждом запуске — только один раз.


### Шаг 3 — поиск через Python
```python
from search import search
import pandas as pd

corpus_clean = pd.read_csv("jokes_clean.csv")
results = search("муж жена дома", corpus_clean, method="bm25", top_k=5)
print(results)
```

### Шаг 4 — поиск через CLI
```bash
python cli.py --query "муж жена дома" --method bm25 --top_k 5 --corpus jokes_clean.csv
```

---
## Параметры CLI

| Параметр | Сокращение | По умолчанию | Описание |
|----------|------------|--------------|----------|
| `--query` | `-q` | обязательный | Текст поискового запроса |
| `--method` | `-m` | `bm25` | Метод поиска: `bm25`, `w2v`, `navec` |
| `--top_k` | `-k` | `5` | Количество результатов |
| `--corpus` | `-c` | `jokes_clean.csv` | Путь к предобработанному CSV-файлу |

### Примеры
```bash
# базовый поиск через BM25
python cli.py -q "врач пациент больница"

# топ-10 через Word2Vec
python cli.py -q "школа учитель урок" -m w2v -k 10

# поиск через Navec
python cli.py -q "кот собака" -m navec -k 5
```

---

## Вывод результатов

Каждый запрос выводит время поиска и пронумерованный список документов с оценкой релевантности:
```
Запрос: «муж жена дома»
Метод: BM25
Топ-3 результатов

  время поиска: 1.4486 сек.
1. Если хозяин в доме - жена, то муж там - хозяйка.
   score: 14.0394

2. Жена - мужу: - Ну какой боулинг? В доме шаром покати!
   score: 13.4972
```

При каждом поиске выводится время выполнения и список документов с оценкой релевантности.

Диапазон оценок метрик:
- BM25: от 0 и выше, чем больше тем релевантнее
- Word2Vec / Navec: косинусное сходство от 0 до 1, где 1 - полное совпадение
