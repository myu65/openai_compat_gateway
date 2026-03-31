from __future__ import annotations

from fastapi import FastAPI

from app.api.chat_completions import router as chat_router

app = FastAPI(title="OpenAI Compat Gateway", version="0.1.0")
app.include_router(chat_router)


@app.get("/healthz")
def healthz():
    return {"ok": True}
