# DeskGuard Base RAG Application

DeskGuard is a support chatbot for a fictional digital banking app. Phase 1 is a baseline for retrieval-augmented generation (RAG), which means finding reference passages before asking a language model to write an answer. The knowledge base contains 50 fictional public support articles covering accounts, cards, transfers, UPI, sign-in, records, and support. The app has no connection to bank accounts.

## Run on Windows

Open PowerShell in the DeskGuard folder. Use Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
.\.venv\Scripts\python.exe app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). The app starts in **Local passage preview**. You can ask questions, see the closest passage, and inspect the retrieval results without a key.

The server uses FastAPI and Uvicorn. Open [API docs](http://127.0.0.1:5000/docs) to try `POST /ask` with a JSON body such as `{"question": "I lost my card. What now?"}`.

To generate answers, set one API key in `.env` and restart the app. If several keys are set, DeskGuard selects the first provider in the table. PowerShell environment variables take precedence over values in the file.

| Key | Model setting | Default model |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | `OPENROUTER_MODEL` | `openrouter/free` |
| `GROQ_API_KEY` | `GROQ_MODEL` | `openai/gpt-oss-20b` |
| `GEMINI_API_KEY` | `GEMINI_MODEL` | `gemini-3.1-flash-lite` |
| `OPENAI_API_KEY` | `OPENAI_MODEL` | `gpt-4o-mini` |

DeskGuard sends the current question and retrieved fictional passages to the selected provider. The chat displays previous turns in the browser and processes each question on its own.

If your browser cannot reach the local server, open the desktop chat instead:

```powershell
.\.venv\Scripts\python.exe desktop.py
```

The Tkinter desktop window calls the shared chat code directly and displays the same answers, sources, and retrieved passages. It works without running the web server.

## Try these questions

| Question | Expected result |
| --- | --- |
| I lost my card. What now? | Find the missing-card section with the steps to freeze it. |
| I can’t find my debit plastic. How do I lock it? | Find the same section from a paraphrase. |
| Money was deducted but my transfer bounced. When will it come back? | Find the 2-business-day return policy for a failed transfer. |
| How long does a merchant refund take? | Find the 5-to-10-business-day timeline once the merchant issues the refund. |
| How do I update my phone number? | Find the profile steps and SMS verification. |
| How do I stop a UPI AutoPay subscription? | Find the steps to revoke a UPI mandate. |
| Can I use my card abroad? | Find the international card controls. |
| I sent money to the wrong person. What can I do? | Find the steps to report a wrong-recipient transfer. |
| Can I export transactions as a CSV for a custom date range? | Find the Activity filter and CSV export steps. |
| What is my current balance? | Explain that DeskGuard has no access to personal account balances. |

## Follow the lost-card question through the code

1. **Split the articles.** In [retrieval.py](retrieval.py), `load_chunks` reads each Markdown article and treats each `##` section as a **chunk**, a short passage about one task. [Freeze a lost or stolen card](articles/lost-card.md) has two sections. The instructions to freeze the card have the stable ID `lost-card/freeze-your-missing-card`.
2. **Embed the passages and question.** An **embedding** represents text as a vector, a list of numerical weights. `terms` maps common banking synonyms to the same words. For “I can’t find my debit plastic. How do I lock it?”, the resulting terms are `missing`, `card`, and `freeze`. The `embed` function uses **TF-IDF**, short for term frequency–inverse document frequency. It gives weight to terms repeated in a passage and more weight to terms that appear in fewer passages. `LocalIndex` saves the passage vectors in `.cache/embeddings.json`. The app rebuilds the cache on restart when the article contents or the term mapping change.
3. **Retrieve up to three chunks.** `LocalIndex.retrieve` searches passages from all 50 articles and ranks them by **cosine similarity**, which compares the direction of two vectors. For these vectors, a larger score means they share more heavily weighted terms. The freezing instructions rank first for the paraphrase. The score alone cannot tell you whether a passage answers the question.
4. **Generate and cite.** `POST /ask` in [app.py](app.py) calls `DeskGuard.ask` in [chat.py](chat.py), which retrieves the matches and passes them to [answering.py](answering.py). `build_model_input` gives the model the question, the passage text and IDs, and instructions to use the passages. The model returns answer text and cited passage IDs. `generate_answer` accepts citations that match the retrieved set. If the passage evidence is insufficient, the expected model response contains no answer or citations, and DeskGuard displays its fallback.
5. **Inspect the result.** The relevant instructions are **Cards → select the missing card → Freeze card → Confirm**. Expand **Retrieved passages** below the reply to see the ranked passages and their scores. Expand **Model input** inside it to see the instructions and question-plus-passages JSON. With the key unset, the inspector labels those fields as prepared input and the reply quotes the closest matching article text. Preview is a way to inspect retrieval and can surface related text for a question that the articles do not answer.

## Verify locally

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The checks exercise the real Markdown corpus, the disk cache, paraphrase retrieval, and the local chat API. The provider tests use simulated HTTP responses to check the outgoing requests, citation handling, and fallback behavior without API credentials.

## RAG baseline evaluation

The evaluator checks 14 answerable and 3 unsupported fictional banking questions against the bundled articles. Run offline passage preview by default, or pass `--provider` to send the same cases to the configured provider. Add `--json` for per-case results.

```powershell
.\.venv\Scripts\python.exe evaluate_rag.py
.\.venv\Scripts\python.exe evaluate_rag.py --provider
```

| Check | Local passage preview | OpenRouter (`openrouter/free`) |
| --- | ---: | ---: |
| Expected passage ranked first | 12/14 | 12/14 |
| Expected passage in the top three | 14/14 | 14/14 |
| Answer contains all checked facts and cites the expected passage | 12/14 | 14/14 |
| Unsupported question gets the fallback without citations | 1/3 | 3/3 |
| Operational errors | 0/17 | 0/17 |

The offline preview missed the top-ranked passage for an unresolved support complaint and a pending transfer. It also returned related excerpts for two unsupported questions. The OpenRouter column is one observed run with an automatically routed model. The automated fact checks cover selected reference facts and cannot establish full answer accuracy.
