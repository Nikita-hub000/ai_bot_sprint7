from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_vectorstore():
    print("Загрузка модели эмбеддингов (BAAI/bge-m3)")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    vectordb = Chroma(
        persist_directory="chroma",
        embedding_function=embeddings
    )

    return vectordb

def normalize_score(score):
    return min(max(score, 0.0), 1.0)

def run_tests(vectordb):
    print("\n=== Проверка качества поиска ===")

    test_queries = [
        {"query": "О чем идет речь в документе 1_kota.txt?", "filter": {"filename": "1_kota.txt"}},
        {"query": "Дай 3 факта о Кутах", "filter": None},
        {"query": "Опиши ключевые факты из файла 6_kota.txt.", "filter": {"filename": "6_kota.txt"}}
    ]

    for item in test_queries:
        query = item["query"]
        filt = item["filter"]
        print("\n" + "=" * 70)
        print(f"Запрос: {query}")

        results = vectordb.similarity_search_with_score(query, k=5, filter=filt)

        seen_chunks = set()
        count = 0
        for doc, score in results:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk_id)
            count += 1

            print(f"\nРезультат {count}")
            print(f"Score : {normalize_score(score):.4f}")
            print(f"Источник: {doc.metadata}")
            print(f"Текст:\n{doc.page_content[:300]}...")

            if count >= 3:
                break


def main():
    vectordb = load_vectorstore()
    run_tests(vectordb)


if __name__ == "__main__":
    main()
