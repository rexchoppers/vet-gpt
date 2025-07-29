import uuid

from dotenv import load_dotenv
from pypdf import PdfReader
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
                # If file ends with .txt, we can skip it
                if file_path.suffix.lower() == ".txt":
                    continue

                print(f"  - File: {file_path.relative_to(procedure_dir)}")

                pdf_reader = PdfReader(str(file_path))
                text = " ".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())

                chunks = [text[i:i + 500] for i in range(0, len(text), 500)]

                for chunk in chunks:
                    embedding = model.encode(chunk).tolist()

                    qdrant.upsert(
                        collection_name="procedures",
                        points=[
                            {
                                "id": str(uuid.uuid4()),
                                "vector": embedding,
                                "payload": {
                                    "text": chunk,
                                    "procedure": procedure_name,
                                    "file_name": file_path.name,
                                    "source": "application/pdf"
                                }
                            }
                        ]
                    )
