"""Deduplication: hash-based exact dedup + embedding-based semantic dedup.

Two layers:
1. Hash dedup (fast): SHA-256(title|source_id|date) — catches exact same finding from same source
2. Semantic dedup (slower): cosine similarity of title+summary embeddings — catches same news
   from different sources with different wording (threshold: 0.92)
"""

import hashlib
import math
import re
from datetime import datetime, timedelta, timezone

import httpx
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Finding

logger = structlog.get_logger()

SEMANTIC_THRESHOLD = 0.92
LOOKBACK_DAYS = 7


async def is_duplicate(db: AsyncSession, dedup_hash: str) -> bool:
    """Check if a finding with this dedup_hash already exists (exact match)."""
    stmt = select(Finding.id).where(Finding.dedup_hash == dedup_hash).limit(1)
    result = await db.execute(stmt)
    exists = result.scalar_one_or_none() is not None

    if exists:
        logger.debug("duplicate_finding_hash", dedup_hash=dedup_hash[:12])

    return exists


async def is_semantic_duplicate(
    db: AsyncSession,
    title: str,
    summary: str,
    source_id: int,
) -> bool:
    """Check if a semantically similar finding exists from a different source.

    Uses embedding cosine similarity to detect the same news reported by
    different sources with different wording.

    Returns True if a finding with similarity > SEMANTIC_THRESHOLD exists
    from a different source within the lookback window.
    """
    # Build text to compare
    new_text = _normalize_text(f"{title} {summary}")
    if not new_text:
        return False

    # Get recent findings from OTHER sources
    since = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    stmt = (
        select(Finding.id, Finding.title, Finding.summary, Finding.source_id)
        .where(
            Finding.created_at >= since,
            Finding.source_id != source_id,
            Finding.status != "dismissed",
        )
        .order_by(Finding.relevance_score.desc())
        .limit(200)
    )
    result = await db.execute(stmt)
    recent = result.all()

    if not recent:
        return False

    # Try embedding-based comparison first, fall back to TF-IDF
    new_embedding = await _get_embedding(new_text)

    if new_embedding is not None:
        # Embedding-based comparison
        for row in recent:
            existing_text = _normalize_text(f"{row.title} {row.summary}")
            existing_embedding = await _get_embedding(existing_text)
            if existing_embedding is not None:
                sim = _cosine_similarity(new_embedding, existing_embedding)
                if sim >= SEMANTIC_THRESHOLD:
                    logger.info(
                        "semantic_duplicate_found",
                        new_title=title[:60],
                        existing_id=row.id,
                        existing_title=row.title[:60],
                        similarity=round(sim, 4),
                    )
                    return True
    else:
        # Fallback: TF-IDF word overlap (no API needed)
        new_tokens = _tokenize(new_text)
        for row in recent:
            existing_text = _normalize_text(f"{row.title} {row.summary}")
            existing_tokens = _tokenize(existing_text)
            sim = _jaccard_similarity(new_tokens, existing_tokens)
            if sim >= 0.65:  # Lower threshold for word overlap
                logger.info(
                    "semantic_duplicate_found_tfidf",
                    new_title=title[:60],
                    existing_id=row.id,
                    existing_title=row.title[:60],
                    similarity=round(sim, 4),
                )
                return True

    return False


# --- Embedding helpers ---

# Simple in-memory cache to avoid re-embedding the same text within a scan
_embedding_cache: dict[str, list[float]] = {}
_CACHE_MAX = 500


async def _get_embedding(text: str) -> list[float] | None:
    """Get embedding vector for text. Returns None if no embedding API available."""
    cache_key = hashlib.md5(text.encode()).hexdigest()
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    # Try OpenAI embeddings API (works with both OpenAI API key and Codex)
    api_key = settings.openai_api_key
    if not api_key:
        return None

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": settings.embedding_model,
                    "input": text[:8000],  # API limit
                },
            )
            resp.raise_for_status()
            data = resp.json()
            embedding = data["data"][0]["embedding"]

            # Cache it
            if len(_embedding_cache) < _CACHE_MAX:
                _embedding_cache[cache_key] = embedding

            return embedding

    except Exception as e:
        logger.debug("embedding_failed", error=str(e))
        return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# --- Fallback: word-level similarity (no API needed) ---


def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Common English stop words to filter out
_STOP_WORDS = frozenset(
    "a an the and or but in on at to for of is it its this that with from by as are was were "
    "be been being have has had do does did will would could should may might can shall not no "
    "so if then than more also very just about up out into over after before between through "
    "during without within along across against until while since when where how what which who "
    "whom their there here these those each every all both few many much some any other another".split()
)


def _tokenize(text: str) -> set[str]:
    """Tokenize text into a set of meaningful words (stop words removed)."""
    words = text.split()
    return {w for w in words if w not in _STOP_WORDS and len(w) > 2}


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    intersection = a & b
    union = a | b
    return len(intersection) / len(union)


def clear_embedding_cache() -> None:
    """Clear the in-memory embedding cache (call between scan runs)."""
    _embedding_cache.clear()
