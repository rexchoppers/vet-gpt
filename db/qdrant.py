import os

from qdrant_client import QdrantClient

qdrant = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", "6333")),
    api_key=os.getenv("QDRANT_API_KEY", None),
    https=os.getenv("QDRANT_HTTPS", "False").lower() == "true",
)