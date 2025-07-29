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
LLM_MODEL_PATH = "./models/Phi-3-mini-128k-instruct.Q4_0.gguf"

llm = Llama(
    model_path=LLM_MODEL_PATH,
    n_ctx=4096,  # Phi-3-mini-128k-instruct supports 128K context. Be mindful of RAM usage.
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

    context_parts = []
    sources = set()

    for hit in search_results.points:
        chunk_text = hit.payload.get("text", "N/A")
        procedure_name = hit.payload.get("procedure", "N/A")
        file_name = hit.payload.get("file_name", "N/A")
        category = hit.payload.get("category", "N/A") # Assuming you added 'category' in ingestion

        # Format context for the LLM, including metadata
        context_parts.append(
            f"--- Document Chunk ---\n"
            f"Category: {category}\n" # Include category if you added it
            f"Item: {procedure_name}\n"
            f"Source File: {file_name}\n"
            f"Content: {chunk_text}"
        )
        sources.add(f"{category or procedure_name} - {file_name}") # Better source tracking

    context_string = "\n\n".join(context_parts)

    system_message = (
        "<|system|>"
        "You are a helpful and knowledgeable veterinary AI assistant. Your goal is to provide accurate and concise answers based *only* on the provided veterinary information. "
        "Each piece of information is clearly labeled with its 'Category', 'Item', and 'Source File'. If the information is insufficient to answer the question, say so and do not invent details. "
        "Always cite the 'Source File' you use.<|end|>"
    )

    user_message = f"""<|user|>
    Carefully read the 'Content' from the relevant documents below, then answer my question.

    --- Veterinary Information ---
    {context_string}
    --- End of Veterinary Information ---

    My Question: {query.q}<|end|>"""

    prompt_template = system_message + user_message + "<|assistant|>"
    print(prompt_template)

    print("[LLM] Sending prompt to LLM...")

    output = llm(
        prompt_template,
        max_tokens=1000,  # Max tokens for the LLM's response
        temperature=0.2,  # Controls creativity (0.0 for deterministic, 1.0 for very creative)
        stop=["<|endoftext|>", "<|end|>"],  # Important stop tokens for Phi-3 models
    )

    print("[LLM] Sent prompt to LLM...")

    print(output["choices"])

    llm_response_text = output["choices"][0]["text"].strip()


    print(llm_response_text)

    return {
        "answer": llm_response_text,
        "sources": list(sources)
    }