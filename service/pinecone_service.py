import repositories.pinecone_repository as prepo

def upsert_file(file_path: str) -> None:
    with open(file_path, "r", encoding="utf-8") as file:
        documents = [line for line in file if line.strip()]
        prepo.upsert_batch(documents)

def upsert(doc: str) -> None:
    prepo.upsert(doc)

def find_similar(query_text, top_k=5):
    return prepo.find_similar(query_text, top_k)