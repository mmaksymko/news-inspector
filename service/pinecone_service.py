import asyncio
from collections.abc import Collection

import repositories.pinecone_repository as prepo
from utils.log_utils import log_io

def upsert_file(file_path: str) -> None:
    with open(file_path, "r", encoding="utf-8") as file:
        documents = [line for line in file if line.strip()]
        prepo.upsert_batch(documents)

def upsert(doc: str) -> None:
    prepo.upsert(doc)

def find_similar(query_text: str, top_k: int = 5):
    return prepo.find_similar(query_text, top_k)

async def find_similar_batch(query_text: Collection, top_k: int = 5):
    return await prepo.find_similar_batch(query_text, top_k)