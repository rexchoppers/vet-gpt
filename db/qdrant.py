"""
Simple singleton connection to Qdrant vector database.

This module provides a ready-to-use instance of the Qdrant client
that can be imported and used throughout the application.

Usage:
    from db.qdrant import qdrant
    
    # Use the client directly
    qdrant.create_collection(...)
    qdrant.search(...)

Note: Requires the qdrant-client package.
Install it using: pip install qdrant-client
"""


import os

from qdrant_client import QdrantClient

qdrant = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", "6333")),
    api_key=os.getenv("QDRANT_API_KEY", None),
    https=os.getenv("QDRANT_HTTPS", "False").lower() == "true",
)