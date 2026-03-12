import os
import time
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA = "knowledge_base"
DIRECTORY = "chroma"
MODEL_NAME = "BAAI/bge-m3"

def load_documents(data_path):
    documents = []
    data_dir = Path(data_path)

    for file_path in data_dir.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(file_path),
                        "filename": file_path.name
                    }
                )
            )
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks

def create_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return embeddings

def build_vectorstore(chunks, embeddings):
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DIRECTORY
    )
    vectordb.persist()
    print("Индекс сохранён в:", DIRECTORY)
    return vectordb

def main():
    start_time = time.time()

    print("=== Загрузка документов ===")
    documents = load_documents(DATA)

    print("\n=== Разбиение на чанки ===")
    chunks = split_documents(documents)

    print("\n=== Инициализация модели эмбеддингов ===")
    embeddings = create_embeddings()

    print("\n=== Создание индекса ===")
    vectordb = build_vectorstore(chunks, embeddings)

    end_time = time.time()
    print(f"\nВремя выполнения: {round(end_time - start_time, 2)} сек.")

if __name__ == "__main__":
    main()
