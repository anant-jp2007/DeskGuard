"""Build the model input, request a grounded answer, and resolve citations."""

import json
from dataclasses import dataclass
from http.client import HTTPException
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from retrieval import Match

INSTRUCTIONS = """You are DeskGuard, a support assistant for a fictional banking app.
Answer the question in plain language using only the supplied support passages.
Keep exact steps, fees, and timelines faithful to the passages.
You can explain public procedures. You cannot see anyone's balance, transactions,
or other personal account details, and you cannot take actions in the app.
If the passages do not support an answer, use an empty answer and no source IDs.
For a supported answer, return the answer as plain text and list the exact IDs
of the passages you relied on in source_ids. Do not put source IDs in the answer."""
GROQ_JSON_INSTRUCTIONS = (
    '\nReturn a JSON object with exactly two fields. "answer" must be a string, '
    'and "source_ids" must be an array of strings.'
)
CHAT_COMPLETION_PROVIDERS = {
    "groq": ("Groq", "https://api.groq.com/openai/v1/chat/completions"),
    "openrouter": ("OpenRouter", "https://openrouter.ai/api/v1/chat/completions"),
    "gemini": ("Gemini", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"),
}

FALLBACK = (
    "I couldn’t find an answer in the public support articles. "
    "DeskGuard can’t access account balances or other personal account details."
)
PREVIEW_MIN_SIMILARITY = 0.20
RESPONSE_FORMAT = {
    "type": "json_schema",
    "name": "deskguard_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "source_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["answer", "source_ids"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class Answer:
    text: str
    source_ids: tuple[str, ...] = ()
    status: str = "fallback"


class ModelError(Exception):
    """An error message that can be shown in the local UI."""


def groq_forbidden_message(error: HTTPError, model: str) -> str:
    try:
        body = error.read(8192)
    except (OSError, HTTPException):
        body = b""
    if body.strip().lower() == b"error code: 1010":
        return "Groq's network gateway blocked the request (HTTP 403, Cloudflare 1010)."

    try:
        code = json.loads(body)["error"]["code"]
    except (ValueError, KeyError, TypeError):
        code = None
    if code == "model_permission_blocked_project":
        return f"Groq blocked {model} for this project (HTTP 403). Check Projects > Limits in the Groq Console."
    if code == "model_permission_blocked_org":
        return f"Groq blocked {model} for this organization (HTTP 403). Check Organization > Limits in the Groq Console."
    return f"Groq denied access to {model} (HTTP 403). Check the project's model permissions in the Groq Console."


def build_model_input(question: str, matches: list[Match], *, provider: str = "openai") -> dict:
    instructions = INSTRUCTIONS + GROQ_JSON_INSTRUCTIONS if provider == "groq" else INSTRUCTIONS
    return {
        "instructions": instructions,
        "input": json.dumps({
            "question": question,
            "passages": [{
                "id": match.chunk.id,
                "title": match.chunk.title,
                "section": match.chunk.section,
                "text": match.chunk.text,
            } for match in matches],
        }, ensure_ascii=False, indent=2),
    }


def request_model(model_input: dict, api_key: str, model: str, *, provider: str = "openai") -> dict:
    if provider in CHAT_COMPLETION_PROVIDERS:
        provider_name, endpoint = CHAT_COMPLETION_PROVIDERS[provider]
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": model_input["instructions"]},
                {"role": "user", "content": model_input["input"]},
            ],
            "response_format": {"type": "json_object"},
        }
        if provider in {"openrouter", "gemini"}:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {key: value for key, value in RESPONSE_FORMAT.items() if key != "type"},
            }
        if provider == "openrouter":
            payload["provider"] = {"require_parameters": True}
    else:
        endpoint = "https://api.openai.com/v1/responses"
        provider_name = "OpenAI"
        payload = {
            **model_input,
            "model": model,
            "store": False,
            "text": {"format": RESPONSE_FORMAT},
        }
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "DeskGuard",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=35) as response:
            result = json.load(response)
    except HTTPError as error:
        with error:
            if error.code == 401:
                raise ModelError(
                    f"{provider_name} rejected the API key (HTTP 401). "
                    f"Check {provider.upper()}_API_KEY and restart DeskGuard."
                ) from error
            if provider == "groq" and error.code == 403:
                raise ModelError(groq_forbidden_message(error, model)) from error
            raise ModelError(f"{provider_name} request failed (HTTP {error.code}). Check your API settings.") from error
    except OSError as error:
        raise ModelError("The model service could not be reached. Try again in a moment.") from error
    except ValueError as error:
        raise ModelError("The model service returned an unreadable response.") from error

    try:
        if provider in CHAT_COMPLETION_PROVIDERS:
            return json.loads(result["choices"][0]["message"]["content"])
        if result["status"] != "completed":
            raise ModelError("The model did not complete its response.")
        for item in result["output"]:
            if item["type"] == "message":
                for part in item["content"]:
                    if part["type"] == "output_text":
                        return json.loads(part["text"])
    except (IndexError, KeyError, ValueError, TypeError) as error:
        raise ModelError("The model service returned an unreadable answer.") from error
    raise ModelError("The model did not return an answer.")


def generate_answer(
    model_input: dict, matches: list[Match], api_key: str, model: str, *, provider: str = "openai"
) -> Answer:
    if not matches:
        return Answer(FALLBACK)

    if not api_key:
        best = matches[0]
        if best.similarity < PREVIEW_MIN_SIMILARITY:
            return Answer(FALLBACK)
        return Answer(best.chunk.text, (best.chunk.id,), "preview")

    try:
        result = request_model(model_input, api_key, model, provider=provider)
    except ModelError as error:
        return Answer(str(error), status="error")

    available_ids = {match.chunk.id for match in matches}
    answer = result.get("answer") if isinstance(result, dict) else None
    source_ids = result.get("source_ids") if isinstance(result, dict) else None
    if (
        isinstance(answer, str) and answer.strip()
        and isinstance(source_ids, list) and source_ids
        and all(isinstance(source_id, str) and source_id in available_ids for source_id in source_ids)
    ):
        return Answer(answer.strip(), tuple(dict.fromkeys(source_ids)), "generated")
    return Answer(FALLBACK)
