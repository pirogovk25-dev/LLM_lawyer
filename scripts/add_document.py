import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, re
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from app.config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, DENSE_MODEL, SPARSE_MODEL

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CHUNKED_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chunked_documents.json")

os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")


def clean_text(text):
    text = re.sub(r'\(в ред\. Федеральны[хго]+ закон[аов]+.*?\)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def parse_docx(file_path):
    doc = Document(file_path)
    return "\n".join(clean_text(p.text) for p in doc.paragraphs if clean_text(p.text))


def parse_all_docs():
    docs = []
    for filename in sorted(os.listdir(DATA_DIR)):
        if filename.endswith(".docx"):
            path = os.path.join(DATA_DIR, filename)
            print(f"  Парсинг: {filename}")
            docs.append({"source": filename, "content": parse_docx(path)})
    return docs


def chunk_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = []
    for doc in documents:
        for i, text in enumerate(splitter.split_text(doc["content"])):
            chunks.append({
                "chunk_id": f"{doc['source']}_{i}",
                "source": doc["source"],
                "text": text,
            })
    return chunks


def reindex(chunks):
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)
    client.set_model(DENSE_MODEL)
    client.set_sparse_model(SPARSE_MODEL)

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Удалена старая коллекция: {COLLECTION_NAME}")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=client.get_fastembed_vector_params(on_disk=False),
        sparse_vectors_config=client.get_fastembed_sparse_vector_params(on_disk=False),
    )
    print(f"  Создана коллекция: {COLLECTION_NAME}")

    batch_size = 50
    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        client.add(
            collection_name=COLLECTION_NAME,
            documents=[c["text"] for c in batch],
            metadata=[{"source": c["source"], "chunk_id": c["chunk_id"]} for c in batch],
            ids=list(range(i, i + len(batch))),
        )
        total += len(batch)
        print(f"  Проиндексировано: {total}/{len(chunks)}")

    print(f"  Готово. Всего в коллекции: {total} чанков.")


def main():
    print("=== Шаг 1: Парсинг документов из data/ ===")
    documents = parse_all_docs()
    print(f"  Найдено файлов: {len(documents)}")

    print("\n=== Шаг 2: Чанкование ===")
    chunks = chunk_docs(documents)
    print(f"  Создано чанков: {len(chunks)}")

    with open(CHUNKED_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"  Сохранено в: {CHUNKED_FILE}")

    print("\n=== Шаг 3: Переиндексация в Qdrant ===")
    reindex(chunks)

    print("\n=== Готово! ===")
    sources = {}
    for c in chunks:
        sources[c["source"]] = sources.get(c["source"], 0) + 1
    for src, cnt in sorted(sources.items()):
        print(f"  {src}: {cnt} чанков")


if __name__ == "__main__":
    main()
