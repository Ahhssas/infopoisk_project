import argparse
import pandas as pd
from preprocessing import preprocess_corpus
from search import search


def main():
    parser = argparse.ArgumentParser(description="Поиск по корпусу анекдотов")
    parser.add_argument("--query", "-q", type=str, required=True, help="Текст запроса")
    parser.add_argument("--method", "-m", type=str, choices=["bm25", "w2v", "navec"],
                        default="bm25", help="Метод поиска (default: bm25)")
    parser.add_argument("--top_k", "-k", type=int, default=5, help="Количество результатов (default: 5)")
    parser.add_argument("--corpus", "-c", type=str, default="/content/jokes_2000.csv",
                        help="Путь к CSV-файлу корпуса")

    args = parser.parse_args()

    df = pd.read_csv(args.corpus)

    if "text_clean" not in df.columns:
        print("Предобработка корпуса...")
        corpus = preprocess_corpus(df, text_col="text")
    else:
        corpus = df

    print(f"\nЗапрос: «{args.query}»")
    print(f"Метод: {args.method.upper()}")
    print(f"Топ-{args.top_k} результатов\n")

    results = search(args.query, corpus, method=args.method, top_k=args.top_k)

    for i, row in results.iterrows():
        text = row["text"].replace("\r", "").replace("\n", " ").strip()
        print(f"{i+1}. {text}")
        print(f" score: {row['score']:.4f}\n")


if __name__ == "__main__":
    main()
