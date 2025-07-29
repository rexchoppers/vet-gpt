import os

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from vllm import LLM
from llama_cpp import Llama


# Load environment variables from a .env file
load_dotenv()

# Import the Qdrant client singleton
from db.qdrant import qdrant

app = FastAPI()

print("[Sentence Transformer] Initializing...")
sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
print("[Sentence Transformer] Initialized.")

print("[LLM] Initializing...")
LLM_MODEL_PATH = "./models/Phi-3-mini-128k-instruct.Q2_K.gguf"

llm = Llama(
    model_path=LLM_MODEL_PATH,
    n_ctx=128000,  # Phi-3-mini-128k-instruct supports 128K context. Be mindful of RAM usage.
    n_gpu_layers=0,  # Set to 0 for CPU-only inference. Set to -1 to offload all layers to GPU if available.
    verbose=False  # Set to True for more detailed loading logs from llama_cpp
)

print(f"[LLM] Llama.cpp LLM Initialized from {LLM_MODEL_PATH}.")


class Query(BaseModel):
    q: str

@app.post("/q")
async def query(query: Query):
    query_embedding = sentence_transformer.encode(query.q).tolist()

    search_results = qdrant.query_points(
        collection_name="procedures",  # Ensure this matches your ingestion collection name
        query=query_embedding,
        limit=5,  # Retrieve top 5 most similar chunks
        with_payload=True,  # Ensure payload (metadata and text) is returned
        # query_filter=qdrant_filter  # Apply the metadata filter
    )

    print(search_results)

    return {"response": f"You asked: {query.q}"}