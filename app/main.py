from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.chat_completions import close_service, router as chat_router, runtime_status


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        yield
    finally:
        await close_service()


app = FastAPI(title="OpenAI Compat Gateway", version="0.1.0", lifespan=lifespan)
app.include_router(chat_router)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    location = first.get("loc") or []
    param = ".".join(str(item) for item in location if item != "body") or None
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": first.get("msg") or "Invalid request",
                "type": "invalid_request_error",
                "param": param,
                "code": None,
            }
        },
    )


@app.exception_handler(HTTPException)
async def http_error_handler(_request: Request, exc: HTTPException):
    error_type = "authentication_error" if exc.status_code in (401, 403) else "invalid_request_error"
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": str(exc.detail),
                "type": error_type,
                "param": None,
                "code": None,
            }
        },
    )


@app.get("/healthz")
async def healthz():
    return {"ok": True, **runtime_status()}
