"""FastAPI application entry — load dotenv before other local imports."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from routes import cv, full_analysis, interview, jd, match, syllabus

BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    provider = os.getenv("LLM_PROVIDER", "").lower().strip()
    if not provider:
        provider = "gemini" if os.getenv("GEMINI_API_KEY") else "openai"
        
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment variables.")
    elif provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to your environment variables.")
    yield


app = FastAPI(title="AI Job Prep", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jd.router)
app.include_router(cv.router)
app.include_router(match.router)
app.include_router(syllabus.router)
app.include_router(interview.router)
app.include_router(full_analysis.router)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")
