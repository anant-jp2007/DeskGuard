"""Markdown chunks, local embeddings, and cosine-similarity search."""

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

# A few phrases that support customers naturally substitute for one another.
ALIASES = {
    "missing": ("lost", "stolen", "misplaced", "gone missing", "can't find", "cannot find"),
    "freeze": ("lock", "block", "frozen", "pause", "disable"),
    "card": ("cards", "debit card", "debit plastic", "plastic"),
    "transfer": ("transfers", "bank transfer", "sent money", "send money", "payment transfer"),
    "failed": ("fail", "rejected", "bounced", "didn't go through", "did not go through"),
    "debited": ("deducted", "charged", "taken"),
    "refund": ("refunds", "money back", "reimbursement", "reversed"),
    "signin": ("sign in", "sign-in", "signing in", "log in", "log-in", "login", "logging in"),
    "password": ("passwords", "passphrase"),
    "forgot": ("forgotten", "can't remember", "cannot remember"),
    "phone": ("mobile", "cellphone", "telephone"),
    "change": ("update", "edit", "updating"),
    "fee": ("fees", "cost", "charges"),
    "purchase": ("purchases", "transaction", "transactions"),
    "days": ("day",),
}
ALIASES_BY_TERM = {alias: term for term, aliases in ALIASES.items() for alias in aliases}
ALIAS_PATTERN = re.compile(
    r"\b(?:" + "|".join(map(re.escape, sorted(ALIASES_BY_TERM, key=len, reverse=True))) + r")\b"
)
STOP_WORDS = set(
    "a an and are as at be been by can could did do does for from get had has have how "
    "i if in is it me my need now of on or our should so that the then there this to "
    "up want was were what when where which who will with would you your current "
    "but s t m re ll d ve".split()
)
INDEX_VERSION = 1


@dataclass(frozen=True)
class Chunk:
    id: str
    article_id: str
    title: str
    section: str
    text: str


@dataclass(frozen=True)
class Match:
    chunk: Chunk
    similarity: float


def load_chunks(articles_dir: Path) -> list[Chunk]:
    """Use each short Markdown section as a passage, preserving its heading."""
    chunks = []
    for path in sorted(articles_dir.glob("*.md")):
        title_line, _, body = path.read_text(encoding="utf-8").partition("\n")
        if not title_line.startswith("# "):
            raise ValueError(f"{path.name} needs a Markdown title on its first line.")
        sections = re.split(r"^## (.+)\n", body, flags=re.MULTILINE)
        if len(sections) == 1:
            raise ValueError(f"{path.name} needs a Markdown section starting with '## '.")
        for section, content in zip(sections[1::2], sections[2::2]):
            slug = re.sub(r"[^a-z0-9]+", "-", section.lower()).strip("-")
            chunks.append(Chunk(
                id=f"{path.stem}/{slug}",
                article_id=path.stem,
                title=title_line[2:].strip(),
                section=section.strip(),
                text=content.strip(),
            ))
    if not chunks:
        raise ValueError(f"No support articles found in {articles_dir}.")
    return chunks


def terms(text: str) -> list[str]:
    text = ALIAS_PATTERN.sub(
        lambda match: ALIASES_BY_TERM[match[0]], text.lower().replace("’", "'")
    )
    return [word for word in re.findall(r"[a-z0-9]+", text) if word not in STOP_WORDS]


def embed(words: list[str], idf: dict[str, float]) -> dict[str, float]:
    """Make a sparse TF-IDF embedding and scale it to unit length."""
    weights = {
        word: (1 + math.log(count)) * idf[word]
        for word, count in Counter(words).items()
        if word in idf
    }
    length = math.sqrt(sum(weight ** 2 for weight in weights.values()))
    return {word: weight / length for word, weight in weights.items()}


class LocalIndex:
    def __init__(self, articles_dir: Path, cache_path: Path):
        self.chunks = load_chunks(articles_dir)
        fingerprint = hashlib.sha256(json.dumps(
            [INDEX_VERSION, ALIASES, sorted(STOP_WORDS), [asdict(chunk) for chunk in self.chunks]],
            ensure_ascii=False,
        ).encode("utf-8")).hexdigest()

        try:
            saved = json.loads(cache_path.read_text(encoding="utf-8"))
            if saved["fingerprint"] == fingerprint and len(saved["vectors"]) == len(self.chunks):
                self.idf = saved["idf"]
                self.vectors = saved["vectors"]
                return
        except (OSError, ValueError, KeyError, TypeError):
            # A missing, stale, or unreadable cache can be rebuilt from the articles.
            pass

        documents = [
            terms(f"{chunk.title}\n{chunk.section}\n{chunk.text}") for chunk in self.chunks
        ]
        document_counts = Counter(word for document in documents for word in set(document))
        self.idf = {
            word: 1 + math.log((1 + len(documents)) / (1 + count))
            for word, count in document_counts.items()
        }
        self.vectors = [embed(document, self.idf) for document in documents]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            mode="w", dir=cache_path.parent, encoding="utf-8", delete=False
        ) as cache_file:
            temp_path = Path(cache_file.name)
            json.dump({
                "fingerprint": fingerprint,
                "idf": self.idf,
                "vectors": self.vectors,
            }, cache_file)
        try:
            temp_path.replace(cache_path)
        finally:
            temp_path.unlink(missing_ok=True)

    def retrieve(self, question: str) -> list[Match]:
        question_vector = embed(terms(question), self.idf)
        # The vectors already have length 1, so their dot product is cosine similarity.
        matches = []
        for chunk, vector in zip(self.chunks, self.vectors):
            similarity = sum(weight * vector.get(word, 0.0) for word, weight in question_vector.items())
            if similarity > 0:
                matches.append(Match(chunk, similarity))
        return sorted(matches, key=lambda match: (-match.similarity, match.chunk.id))[:3]
