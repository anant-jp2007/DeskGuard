"""Serve the DeskGuard chat UI and API with FastAPI."""

import json

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from chat import DeskGuard, InvalidQuestion, QUESTION_MAX_LENGTH, ROOT

MAX_REQUEST_BYTES = 16_384
QUESTION_BODY = {
    "required": True,
    "content": {
        "application/json": {
            "schema": {
                "type": "object",
                "required": ["question"],
                "properties": {
                    "question": {
                        "type": "string",
                        "description": f"1 to {QUESTION_MAX_LENGTH:,} characters after trimming whitespace.",
                    },
                },
            },
            "example": {"question": "I lost my card. What now?"},
        },
    },
}


def create_app(chat: DeskGuard | None = None) -> FastAPI:
    chat = DeskGuard() if chat is None else chat
    app = FastAPI(title="DeskGuard API", redirect_slashes=False)
    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
    templates = Jinja2Templates(directory=ROOT / "templates")

    @app.head("/", include_in_schema=False)
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home(request: Request):
        return templates.TemplateResponse(request, "chat.html", {
            "model_enabled": chat.model_enabled,
            "article_count": chat.article_count,
        })

    @app.post(
        "/ask",
        openapi_extra={"requestBody": QUESTION_BODY},
        responses={
            400: {"description": "Invalid question"},
            413: {"description": "Request body exceeds 16 KiB"},
        },
    )
    async def ask(request: Request):
        # Enforce the byte limit before buffering the entire request.
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_REQUEST_BYTES:
                return JSONResponse({"error": "Request body is too large."}, status_code=413)
            body.extend(chunk)

        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        data = None
        if content_type == "application/json" or (
            content_type.startswith("application/") and content_type.endswith("+json")
        ):
            try:
                data = json.loads(body)
            except (ValueError, RecursionError):
                pass

        question = data.get("question") if isinstance(data, dict) else None
        try:
            return await run_in_threadpool(chat.ask, question)
        except InvalidQuestion as error:
            return JSONResponse({"error": str(error)}, status_code=400)

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=5000)
