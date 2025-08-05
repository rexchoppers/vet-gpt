# Vet-GPT

A veterinary assistant application that uses AI to answer questions based on veterinary information.

## Overview

Vet-GPT combines:
- A FastAPI backend for handling queries
- Vector database (Qdrant) for efficient information retrieval
- LLM (Phi-3-mini) for generating accurate responses

## Quick Start

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Start the Qdrant vector database:
   ```
   docker-compose up -d
   ```

3. Run the application:
   ```
   python main.py
   ```

4. Send queries to the API endpoint at `/q`

## Project Structure

- `main.py`: FastAPI application for handling queries
- `ingest/`: Scripts for ingesting veterinary data
- `db/`: Database connection utilities
- `models/`: LLM model files
- `resources/`: Veterinary information resources