import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import MAX_REQUEST_BYTES, create_app
from chat import DeskGuard, QUESTION_ERROR, QUESTION_MAX_LENGTH, ROOT
from retrieval import LocalIndex

ARTICLE_COUNT = 50
LOST_CARD_ID = "lost-card/freeze-your-missing-card"
PARAPHRASE = "I can’t find my debit plastic. How do I lock it?"


def model_reply(answer, source_ids):
    return io.BytesIO(json.dumps({
        "status": "completed",
        "output": [
            {"type": "message", "content": [{
                "type": "output_text",
                "text": json.dumps({"answer": answer, "source_ids": source_ids}),
            }]}
        ],
    }).encode("utf-8"))


class DeskGuardTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cache = Path(directory.name) / "embeddings.json"
        self.client = self.make_client()

    def make_client(self, api_key=""):
        chat = DeskGuard(api_key=api_key, cache_path=self.cache)
        client = TestClient(create_app(chat))
        self.addCleanup(client.close)
        return client

    def test_corpus_and_embedding_cache(self):
        saved = json.loads(self.cache.read_text(encoding="utf-8"))
        with patch("retrieval.embed", side_effect=AssertionError("Rebuilt cached embeddings")):
            index = LocalIndex(ROOT / "articles", self.cache)
        self.assertEqual(len({chunk.article_id for chunk in index.chunks}), ARTICLE_COUNT)
        self.assertEqual(len(index.chunks), 100)
        self.assertEqual(len({chunk.id for chunk in index.chunks}), len(index.chunks))
        self.assertEqual(index.vectors, saved["vectors"])
        self.assertEqual(len(index.vectors), len(index.chunks))

    def test_cache_rebuilds_when_an_article_changes(self):
        articles = self.cache.parent / "articles"
        articles.mkdir()
        article = articles / "test.md"
        article.write_text("# Test article\n\n## Test section\n\nOriginal text.", encoding="utf-8")
        original = LocalIndex(articles, self.cache)
        article.write_text("# Test article\n\n## Test section\n\nRevised text.", encoding="utf-8")
        revised = LocalIndex(articles, self.cache)
        self.assertIn("original", original.vectors[0])
        self.assertIn("revised", revised.vectors[0])

    def test_chat_page_and_paraphrase_retrieval(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn(f"{ARTICLE_COUNT} public support articles", page.text)
        self.assertIn("Local passage preview", page.text)
        for path in ("/static/chat.js?v=3", "/static/chat.css?v=2"):
            self.assertIn(f'"{path}"', page.text)
            self.assertEqual(self.client.get(path).status_code, 200)

        docs = self.client.get("/docs")
        self.assertEqual(docs.status_code, 200)
        schema = self.client.get("/openapi.json").json()
        body = schema["paths"]["/ask"]["post"]["requestBody"]
        self.assertEqual(body["content"]["application/json"]["schema"]["required"], ["question"])

        for question in ("I lost my card. What now?", PARAPHRASE):
            with self.subTest(question=question):
                response = self.client.post("/ask", json={"question": question})
                self.assertEqual(response.status_code, 200)
                data = response.json()
                matches = data["retrieved"]
                self.assertEqual(data["status"], "preview")
                self.assertEqual(matches[0]["chunk"]["id"], LOST_CARD_ID)
                self.assertLessEqual(len(matches), 3)
                scores = [match["similarity"] for match in matches]
                self.assertEqual(scores, sorted(scores, reverse=True))
                self.assertIn("Freeze card > Confirm", data["answer"])
                self.assertEqual([source["id"] for source in data["sources"]], [LOST_CARD_ID])
                self.assertFalse(data["model_requested"])

    def test_invalid_questions_keep_the_json_error(self):
        for payload in (None, [], {}, {"question": 42}, {"question": "  "},
                        {"question": "x" * (QUESTION_MAX_LENGTH + 1)}):
            with self.subTest(payload=payload):
                response = self.client.post("/ask", json=payload)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json(), {"error": QUESTION_ERROR})

        response = self.client.post(
            "/ask", content=b'{"question":', headers={"Content-Type": "application/json"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error": QUESTION_ERROR})

    def test_request_body_limit(self):
        body = b'{"question":"Hello","padding":""}'
        body += b" " * (MAX_REQUEST_BYTES - len(body))
        headers = {"Content-Type": "application/json"}
        response = self.client.post("/ask", content=body, headers=headers)
        self.assertEqual(response.status_code, 200)
        response = self.client.post("/ask", content=body + b" ", headers=headers)
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json(), {"error": "Request body is too large."})

    def test_preview_selects_the_right_article_and_section(self):
        cases = (
            ("Money was deducted but my transfer bounced. When will it come back?",
             "failed-transfer/money-debited-for-a-failed-transfer", "within 2 business days"),
            ("I sent money to the wrong person. What can I do?",
             "wrong-recipient-transfer/get-help-after-sending-money-to-the-wrong-recipient",
             "Wrong recipient"),
            ("How do I stop a UPI AutoPay subscription?",
             "upi-mandate/cancel-a-subscription-mandate", "Revoke mandate"),
            ("I forgot my UPI PIN. How do I reset it?",
             "upi-pin/set-a-new-upi-pin", "Reset UPI PIN"),
            ("Can I use my card abroad?",
             "card-international-payments/use-your-card-abroad", "International use"),
            ("My card is about to expire. What happens now?",
             "card-expiry/check-renewal-when-your-card-is-about-to-expire", "Renewal"),
            ("Can I export transactions as a CSV for a custom date range?",
             "export-activity/download-activity-for-a-custom-date-range", "Export CSV"),
        )
        for question, source_id, expected_answer in cases:
            with self.subTest(question=question):
                response = self.client.post("/ask", json={"question": question})
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data["status"], "preview")
                self.assertEqual([source["id"] for source in data["sources"]], [source_id])
                self.assertIn(expected_answer, data["answer"])
                self.assertLessEqual(len(data["retrieved"]), 3)

    def test_balance_falls_back_without_calling_the_model(self):
        with patch("answering.urlopen", side_effect=AssertionError("Called the model unexpectedly")):
            data = self.make_client("test-key").post(
                "/ask", json={"question": "What is my current balance?"}
            ).json()
        self.assertEqual(data["status"], "fallback")
        self.assertIn("can’t access account balances", data["answer"])
        self.assertEqual(data["sources"], [])
        self.assertEqual(data["retrieved"], [])

    def test_generated_answer_and_displayed_context(self):
        client = self.make_client("test-key")
        with patch("answering.urlopen", return_value=model_reply(
            "Open Cards, select the missing card, choose Freeze card, and confirm.",
            [LOST_CARD_ID],
        )) as call:
            data = client.post("/ask", json={"question": PARAPHRASE}).json()

        sent = json.loads(call.call_args.args[0].data)
        context = json.loads(sent["input"])
        self.assertEqual(context["question"], PARAPHRASE)
        self.assertEqual(
            [passage["id"] for passage in context["passages"]],
            [match["chunk"]["id"] for match in data["retrieved"]],
        )
        self.assertEqual(sent["instructions"], data["model_input"]["instructions"])
        self.assertEqual(sent["input"], data["model_input"]["input"])
        self.assertNotIn("test-key", json.dumps(data))
        self.assertEqual(data["status"], "generated")
        self.assertTrue(data["model_requested"])
        self.assertEqual([source["id"] for source in data["sources"]], [LOST_CARD_ID])

    def test_related_question_falls_back_with_no_valid_citations(self):
        client = self.make_client("test-key")
        for answer, source_ids in [("", []), ("Unsupported claim", ["unretrieved-id"])]:
            with self.subTest(source_ids=source_ids):
                with patch("answering.urlopen", return_value=model_reply(answer, source_ids)):
                    data = client.post(
                        "/ask", json={"question": "Can I GPS-track my missing card?"}
                    ).json()
                self.assertTrue(data["retrieved"])
                self.assertEqual(data["status"], "fallback")
                self.assertEqual(data["sources"], [])

    def test_groq_uses_its_own_key_and_returns_a_cited_answer(self):
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "",
            "GROQ_API_KEY": "gsk_test-key",
            "GROQ_MODEL": "groq-test-model",
            "OPENAI_API_KEY": "sk-other-provider",
        }):
            client = self.make_client(api_key=None)
        content = json.dumps({"answer": "Use Freeze card in the app.", "source_ids": [LOST_CARD_ID]})
        response = io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}]}).encode())
        with patch("answering.urlopen", return_value=response) as call:
            data = client.post("/ask", json={"question": PARAPHRASE}).json()

        request = call.call_args.args[0]
        sent = json.loads(request.data)
        self.assertEqual(request.full_url, "https://api.groq.com/openai/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer gsk_test-key")
        self.assertEqual(request.get_header("User-agent"), "DeskGuard")
        self.assertEqual(sent["model"], "groq-test-model")
        self.assertEqual(sent["response_format"], {"type": "json_object"})
        self.assertEqual(sent["messages"][0]["content"], data["model_input"]["instructions"])
        self.assertEqual(sent["messages"][1]["content"], data["model_input"]["input"])
        self.assertEqual(data["status"], "generated")
        self.assertEqual([source["id"] for source in data["sources"]], [LOST_CARD_ID])
        self.assertNotIn("gsk_test-key", json.dumps(data))

    def test_openrouter_key_reaches_openrouter_under_either_setting(self):
        for variable in ("OPENROUTER_API_KEY", "GROQ_API_KEY"):
            with self.subTest(variable=variable):
                settings = {
                    "OPENROUTER_API_KEY": "",
                    "OPENROUTER_MODEL": "",
                    "GROQ_API_KEY": "gsk_other-provider",
                    "GROQ_MODEL": "groq-test-model",
                }
                settings[variable] = "sk-or-test"
                with patch.dict(os.environ, settings):
                    client = self.make_client(api_key=None)
                content = json.dumps({"answer": "Freeze the card.", "source_ids": [LOST_CARD_ID]})
                response = io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}]}).encode())
                with patch("answering.urlopen", return_value=response) as call:
                    data = client.post("/ask", json={"question": PARAPHRASE}).json()

                request = call.call_args.args[0]
                payload = json.loads(request.data)
                self.assertEqual(request.full_url, "https://openrouter.ai/api/v1/chat/completions")
                self.assertEqual(request.get_header("Authorization"), "Bearer sk-or-test")
                self.assertEqual(payload["model"], "openrouter/free")
                self.assertEqual(payload["response_format"]["type"], "json_schema")
                self.assertEqual(payload["provider"], {"require_parameters": True})
                self.assertEqual(payload["messages"][1]["content"], data["model_input"]["input"])
                self.assertEqual(data["status"], "generated")
                self.assertEqual([source["id"] for source in data["sources"]], [LOST_CARD_ID])
                self.assertNotIn("sk-or-test", json.dumps(data))

    def test_gemini_uses_its_own_key_and_returns_a_cited_answer(self):
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "",
            "GROQ_API_KEY": "",
            "GEMINI_API_KEY": "gemini-test-key",
            "GEMINI_MODEL": "",
            "OPENAI_API_KEY": "sk-other-provider",
        }):
            client = self.make_client(api_key=None)
        content = json.dumps({"answer": "Freeze the card.", "source_ids": [LOST_CARD_ID]})
        response = io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}]}).encode())
        with patch("answering.urlopen", return_value=response) as call:
            data = client.post("/ask", json={"question": PARAPHRASE}).json()

        request = call.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer gemini-test-key")
        self.assertEqual(payload["model"], "gemini-3.1-flash-lite")
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertEqual(payload["messages"][0]["content"], data["model_input"]["instructions"])
        self.assertEqual(payload["messages"][1]["content"], data["model_input"]["input"])
        self.assertEqual(data["status"], "generated")
        self.assertEqual([source["id"] for source in data["sources"]], [LOST_CARD_ID])
        self.assertNotIn("gemini-test-key", json.dumps(data))

    def test_authentication_error_names_the_provider_setting(self):
        for api_key, provider, setting in (
            ("test-openai-key", "OpenAI", "OPENAI_API_KEY"),
            ("gsk_test-groq-key", "Groq", "GROQ_API_KEY"),
            ("sk-or-test-openrouter-key", "OpenRouter", "OPENROUTER_API_KEY"),
        ):
            with self.subTest(provider=provider):
                error = HTTPError("", 401, "Unauthorized", None, io.BytesIO())
                with patch("answering.urlopen", side_effect=error):
                    data = self.make_client(api_key).post("/ask", json={"question": PARAPHRASE}).json()
                self.assertEqual(data["status"], "error")
                self.assertIn(f"{provider} rejected", data["answer"])
                self.assertIn(setting, data["answer"])
                self.assertNotIn(api_key, json.dumps(data))

    def test_groq_forbidden_error_identifies_known_causes(self):
        client = self.make_client("gsk_test-groq-key")
        for body, expected in (
            (b'{"error":{"code":"model_permission_blocked_project"}}', "Projects > Limits"),
            (b'{"error":{"code":"model_permission_blocked_org"}}', "Organization > Limits"),
            (b"error code: 1010", "Cloudflare 1010"),
            (b'{"error":{"code":"unknown","message":"PRIVATE_UPSTREAM_TEXT"}}', "denied access"),
        ):
            with self.subTest(expected=expected):
                error = HTTPError("", 403, "Forbidden", None, io.BytesIO(body))
                with patch("answering.urlopen", side_effect=error):
                    data = client.post("/ask", json={"question": PARAPHRASE}).json()
                self.assertEqual(data["status"], "error")
                self.assertIn("HTTP 403", data["answer"])
                self.assertIn(expected, data["answer"])
                self.assertNotIn("PRIVATE_UPSTREAM_TEXT", data["answer"])


if __name__ == "__main__":
    unittest.main()
