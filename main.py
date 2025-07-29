from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    q: str


@app.post("/q")
async def query(query: Query):
    return {"response": f"You asked: {query.q}"}