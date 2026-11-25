"""Measure DeskGuard on a fixed set of fictional banking support questions."""

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from chat import DEFAULT_MODELS, DeskGuard


@dataclass(frozen=True)
class Case:
    name: str
    question: str
    source: str = ""
    facts: tuple[tuple[str, str], ...] = ()


CASES = (
    Case("lost-card", "I can't find my debit card. How can I lock it?", "lost-card/freeze-your-missing-card", (
        ("freeze the missing card", r"\bfreeze\b"),
        ("confirm the action", r"\bconfirm\b"),
    )),
    Case("card-abroad", "Where do I turn on card payments for my trip abroad?", "card-international-payments/use-your-card-abroad", (
        ("open card controls", r"\bcontrols\b"),
        ("enable international use", r"\binternational use\b"),
    )),
    Case("failed-transfer", "My transfer failed but my account was charged. When does the money come back?", "failed-transfer/money-debited-for-a-failed-transfer", (
        ("funds are returned automatically", r"\bautomat(?:ic|ically)\b"),
        ("within two business days", r"\b(?:2|two) business days\b"),
    )),
    Case("merchant-refund", "The merchant just issued a refund. How long before I see it in the app?", "merchant-refund/when-a-merchant-refund-arrives", (
        ("five to ten business days", r"\b(?:5|five)\s*(?:-|to)\s*(?:10|ten) business days\b"),
        ("the merchant's issue date starts the window", r"\bmerchant(?:'s|s')? (?:issues|issued|refund date)\b"),
    )),
    Case("new-phone", "I have a new mobile number. Where do I update it and how do I verify it?", "change-phone/change-your-registered-mobile-number", (
        ("profile contact details", r"\bcontact details\b"),
        ("SMS verification on the new number", r"\b(?:sms|text message)\b.*\b(?:new|that) (?:mobile |phone )?number\b"),
    )),
    Case("stop-autopay", "How do I stop future debits for a UPI AutoPay subscription?", "upi-mandate/cancel-a-subscription-mandate", (
        ("open UPI AutoPay", r"\bautopay\b"),
        ("revoke the mandate", r"\brevoke(?: the| your)? mandate\b"),
    )),
    Case("activity-export", "Can I download a CSV of my activity between two dates I pick?", "export-activity/download-activity-for-a-custom-date-range", (
        ("choose a start and end date", r"\b(?:start and end dates?|date range)\b"),
        ("use CSV export", r"\b(?:export csv|(?:download|export).*?\bcsv)\b"),
    )),
    Case("card-pin-lock", "My card PIN was entered incorrectly three times. How do I get help?", "card-pin/too-many-wrong-pin-attempts", (
        ("open the Card PIN issue option", r"\bcard pin issue\b"),
        ("request an identity check", r"\bidentity (?:check|verification)\b"),
    )),
    Case("plan-fees", "Is there a monthly fee for Standard, and what does Plus cost?", "monthly-fee/standard-and-plus-monthly-charges", (
        ("Standard has no monthly fee", r"\bstandard\b[^.]*\b(?:free|no monthly (?:fee|charge))\b"),
        ("Plus costs 99 rupees per month", r"(?:₹\s*99\b|\b(?:inr|rs\.?)\s*99\b|\b99\s*(?:rupees|inr)\b)[^.]*\b(?:per month|a month|monthly)\b"),
    )),
    Case("interest-certificate", "Where can I download last year's interest certificate as a PDF?", "tax-interest-certificate/get-the-annual-interest-certificate-for-tax-records", (
        ("the Documents screen has Interest certificate", r"\bdocuments\b.*\binterest certificate\b"),
        ("select the completed financial year", r"\b(?:completed|previous|last) financial year\b"),
    )),
    Case("complaint", "Support hasn't resolved my case. Where can I file a formal complaint?", "support-complaint/escalate-an-unresolved-support-case", (
        ("open New complaint", r"\bnew complaint\b"),
        ("include the existing case reference", r"\bcase reference\b"),
    )),
    Case("daily-transfer-limit", "What is the domestic transfer limit per day, and when does it reset?", "transfer-limit/daily-transfer-cap", (
        ("50,000 rupees daily", r"(?:₹\s*50[,\s]?000\b|\b(?:inr|rs\.?)\s*50[,\s]?000\b|\b50[,\s]?000\s*(?:rupees|inr)\b)"),
        ("midnight India Standard Time", r"\bmidnight\b[^.]*\b(?:india standard time|ist)\b"),
    )),
    Case("pending-transfer", "My bank transfer is still pending. How long should I wait and should I send it again?", "pending-transfer/wait-for-a-pending-transfer-to-clear", (
        ("up to one business day", r"\b(?:1|one) business day\b"),
        ("avoid a second payment while pending", r"\b(?:avoid|do not|don’t|don't)\b[^.]*\b(?:second|another|again|resending|repeating)\b"),
    )),
    Case("rejected-id-photo", "My ID photo was rejected. How should I retake it?", "identity-verification/retake-a-rejected-id-photo", (
        ("use Retake document photo", r"\bretake document photo\b"),
        ("show all four corners", r"\b(?:all )?(?:4|four) corners\b"),
    )),
    Case("private-balance", "What is my current balance?"),
    Case("unsupported-card-tracking", "Can I track my missing card using GPS?"),
    Case("unsupported-statement-language", "Can I get my monthly account statement in French?"),
)


def normalize_answer(answer: str) -> str:
    answer = unicodedata.normalize("NFKC", answer).casefold()
    return re.sub(r"\s+", " ", answer.translate(str.maketrans("–—−’", "---'")))


def evaluate(chat: DeskGuard) -> dict:
    """Quit provider calls on an operational error and finish local retrieval scoring."""
    rows = []
    provider_failed = False
    for case in CASES:
        row = {"case": case.name, "expected_source": case.source or None}
        if provider_failed:
            row["status"] = "skipped"
            row["retrieved"] = [match.chunk.id for match in chat.index.retrieve(case.question)]
        else:
            reply = chat.ask(case.question)
            row["status"] = reply["status"]
            row["retrieved"] = [match["chunk"]["id"] for match in reply["retrieved"]]
            if reply["status"] == "error":
                provider_failed = True
            else:
                citations = [source["id"] for source in reply["sources"]]
                row["citations"] = citations
                if case.source:
                    answer = normalize_answer(reply["answer"])
                    row["missing_facts"] = [
                        label for label, pattern in case.facts if not re.search(pattern, answer)
                    ]
                else:
                    row["abstained"] = reply["status"] == "fallback" and not citations
        rows.append(row)

    answerable = [row for row in rows if row["expected_source"]]
    answered = [row for row in answerable if "citations" in row]
    abstentions = [row for row in rows if "abstained" in row]
    expected_status = "generated" if chat.model_enabled else "preview"
    metrics = {
        "retrieval_hit_at_1": (sum(row["expected_source"] in row["retrieved"][:1] for row in answerable), len(answerable)),
        "retrieval_hit_at_3": (sum(row["expected_source"] in row["retrieved"][:3] for row in answerable), len(answerable)),
        "expected_citation_present": (sum(row["expected_source"] in row["citations"] for row in answered), len(answered)),
        "only_expected_citation": (sum(row["citations"] == [row["expected_source"]] for row in answered), len(answered)),
        "all_reference_fact_checks": (sum(not row["missing_facts"] for row in answered), len(answered)),
        "answer_and_citation_pass": (
            sum(row["status"] == expected_status and not row["missing_facts"]
                and row["expected_source"] in row["citations"] for row in answered),
            len(answered),
        ),
        "abstention_pass": (sum(row["abstained"] for row in abstentions), len(abstentions)),
        "operational_errors": (sum(row["status"] == "error" for row in rows), len(rows)),
        "skipped_responses": (sum(row["status"] == "skipped" for row in rows), len(rows)),
    }
    result = {
        "mode": "provider" if chat.model_enabled else "preview",
        "metrics": {name: {"count": count, "total": total} for name, (count, total) in metrics.items()},
        "cases": rows,
    }
    if chat.model_enabled:
        result["provider"] = chat.provider
        default_model = DEFAULT_MODELS[chat.provider]
        result["model"] = default_model if chat.model == default_model else "custom"
    return result


def print_report(result: dict) -> None:
    label = result["mode"]
    if label == "provider":
        label += f" ({result['provider']}, {result['model']})"
    print(f"DeskGuard baseline — {label}")
    for name, metric in result["metrics"].items():
        total = metric["total"]
        value = f"{metric['count']}/{total}" if total else "unavailable"
        print(f"{name.replace('_', ' ')}: {value}")
    print()
    for row in result["cases"]:
        expected = row["expected_source"]
        notes = []
        if expected and expected not in row["retrieved"][:1]:
            notes.append("expected source not ranked first")
        if "missing_facts" in row and expected not in row["citations"]:
            notes.append("expected source not cited")
        if row.get("missing_facts"):
            notes.append("missing checks: " + ", ".join(row["missing_facts"]))
        if row.get("abstained") is False:
            notes.append("expected abstention")
        suffix = " — " + "; ".join(notes) if notes else ""
        print(f"{row['case']}: {row['status']}{suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", action="store_true", help="Send the fictional cases to the configured provider.")
    parser.add_argument("--json", action="store_true", help="Print the evaluation as JSON.")
    args = parser.parse_args()
    with TemporaryDirectory(prefix="deskguard-eval-") as directory:
        chat = DeskGuard(api_key=None if args.provider else "", cache_path=Path(directory) / "index.json")
        if args.provider and not chat.model_enabled:
            parser.error("Set a provider API key before running provider evaluation.")
        result = evaluate(chat)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result)
    return int(result["metrics"]["operational_errors"]["count"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
