from dotenv import load_dotenv
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")

# Iterate through documents in ../resources/procedures


import os
import sys
from pathlib import Path

# Add the project root to the Python path to allow importing from sibling directories
current_dir = Path(__file__).parent
project_root = current_dir.parent.resolve()
sys.path.insert(0, str(project_root))

# Now we can import from the db module
from db.qdrant import qdrant

# Get the absolute path to the procedures directory
current_dir = Path(__file__).parent
procedures_dir = (current_dir / ".." / "resources" / "procedures").resolve()

# Check if collection exists
if not qdrant.collection_exists("procedures"):
    print("Creating 'procedures' collection in Qdrant...")
    qdrant.create_collection(
        collection_name="procedures",
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE,
        )
    )
else:
    print("'procedures' collection already exists in Qdrant.")

# Iterate through each procedure directory
for procedure_dir in procedures_dir.iterdir():
    if procedure_dir.is_dir():
        procedure_name = procedure_dir.name
        print(f"Found procedure: {procedure_name}")

        # Iterate through files in the procedure directory
        for file_path in procedure_dir.glob("**/*"):
            if file_path.is_file():
                print(f"  - File: {file_path.relative_to(procedure_dir)}")

# Here’s the roadmap you’re on:
#
# Step	What You're Doing	Status
# 1	FastAPI basic API	✅ Done
# 2	Ingest vet PDF into Qdrant	✅ In progress
# 3	Add embedding search to /ask	🔜 Next
# 4	Connect to model (OpenAI or vLLM)	🔜
# 5	Prompt construction (context + user question)	🔜
# 6	Logging: token usage, latency	🔜
# 7	Auth + optional UI	🔜
#
# Sorry so let's go back to this
#
# How does the model know what is an illness or procedure? Or will it not know or do I have to add metadata to this
