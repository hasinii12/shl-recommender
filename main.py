"""
SHL Assessment Recommender — FastAPI Service
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import logging
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from agent import chat
from retrieval import initialize_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing FAISS index …")
    initialize_index()
    logger.info("Service ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="SHL Assessment Recommender", version="1.0.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1, max_length=50)

    @field_validator("messages")
    @classmethod
    def validate_conversation(cls, msgs):
        if msgs[0].role != "user":
            raise ValueError("Conversation must start with a user message.")
        if len(msgs) > 8:
            raise ValueError("Conversation exceeds the maximum of 8 turns.")
        return msgs


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    try:
        result = chat(messages)
    except Exception as e:
        logger.exception("Unhandled error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error.")

    return ChatResponse(
        reply=result["reply"],
        recommendations=[Recommendation(**r) for r in result.get("recommendations", [])],
        end_of_conversation=bool(result.get("end_of_conversation", False)),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": "An unexpected error occurred."})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)