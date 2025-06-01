import asyncio
import logging
import os
from uuid import uuid4
import numpy as np
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModel
import torch

INDEX_NAME = os.getenv('PINECONE_INDEX')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_MODEL_NAME = os.getenv('PINECONE_MODEL_NAME')

cone = Pinecone(api_key=PINECONE_API_KEY, environment="us-east-1-aws")
index = cone.Index(INDEX_NAME)
host = cone.describe_index(INDEX_NAME).host
tokenizer = AutoTokenizer.from_pretrained(PINECONE_MODEL_NAME)
model = AutoModel.from_pretrained(PINECONE_MODEL_NAME)

def encode_text(text: str) -> np.ndarray:
    """
    Encodes a list of texts into 1024-dimensional vectors using the multilingual-e5-large model.
    """
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Average pooling
    return embeddings.numpy()

def get_vector(line: str) -> dict[str,str|np.ndarray]:
    id = str(uuid4())
    text = line.strip()
    embedding = encode_text([text])[0]
    return {"id": id, "values": embedding.tolist(), "metadata": {"text": text}}

def upsert(text: str) -> None:
    upsert_batch([text])

def upsert_batch(documents) -> None:
    vectors = [get_vector(doc) for doc in documents]
    index.upsert(vectors)
    logging.info("Upsert completed successfully!")

def find_similar(query_text, top_k):
    """
    Finds the top K most similar documents to the query text.
    """
    query_embedding = encode_text([query_text])[0]
    
    query_results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )
    
    logging.info(f"Query: {query_text}")
    logging.info(f"Matches: {query_results['matches']}")
        
    return extract_texts_from_results([query_results])

async def find_similar_batch(claims: list[str], top_k: int = 5) -> list[list[str]]:
    """
    Finds the top K most similar documents for each string in 'claims',
    using Pinecone’s AsyncIO client. Returns a list-of-lists of metadata['text'].
    """
    # 1) get the Pinecone host
    host = cone.describe_index(INDEX_NAME).host

    # 2) open the IndexAsyncio client here—and do NOT exit until after all queries are done
    async with cone.IndexAsyncio(host=host) as async_index:
        async def query_one(claim: str) -> list[str]:
            # Note: encode_text expects a list, so we pass [claim] and then grab [0]
            vec = encode_text([claim])[0].tolist()
            response = await async_index.query(
                vector=vec,
                top_k=top_k,
                include_metadata=True,
            )
            return extract_texts_from_results([response])
        tasks = [query_one(c) for c in claims]
        results: list[list[str]] = await asyncio.gather(*tasks)
    return results


def extract_texts_from_results(results):
    """
    Extracts texts from the results returned by Pinecone.
    """
# Using a list comprehension to simplify the code
    print(f"Extracting texts from results: {results}")
    result =  [match['metadata']['text'] for result in results for match in result['matches']]
    print(f"Extracted texts: {result}")
    return result
    