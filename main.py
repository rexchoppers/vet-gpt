from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

# Load environment variables from a .env file
load_dotenv()

app = FastAPI()

class Query(BaseModel):
    q: str

@app.post("/q")
async def query(query: Query):
    return {"response": f"You asked: {query.q}"}