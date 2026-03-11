"""Relevance scoring for findings against Stratum's verticals and thesis."""

import hashlib
from datetime import date

import structlog

from app.config import settings
from app.models import Source

logger = structlog.get_logger()

# Reference terms for vertical alignment scoring
VERTICAL_KEYWORDS: dict[str, list[str]] = {
    "identity_permissioning": [
        "identity", "eid", "eidas", "verifiable credential", "did", "decentralized id",
        "access control", "permissioning", "ssi", "self-sovereign", "kyc identity",
        "digital identity", "zero knowledge proof", "zk-proof",
    ],
    "wallets_key_management": [
        "wallet", "custody", "mpc", "key management", "multi-party computation",
        "institutional wallet", "cold storage", "hardware security module", "hsm",
        "account abstraction", "smart account",
    ],
    "compliance_trust": [
        "kyc", "kyb", "aml", "sanctions", "compliance", "regtech", "fraud detection",
        "transaction monitoring", "mica", "dlt pilot", "regulatory", "anti-money laundering",
        "travel rule", "fatf",
    ],
    "data_oracles_middleware": [
        "oracle", "data feed", "pricing", "valuation", "tokenisation", "tokenization",
        "middleware", "interoperability", "cross-chain", "bridge", "settlement",
        "messaging", "api", "integration layer",
    ],
}

# Geographic relevance keywords
EUROPE_KEYWORDS = [
    "europe", "european", "eu", "mica", "dlt pilot", "esma", "bafin", "fca",
    "switzerland", "swiss", "uk", "germany", "france", "sweden", "nordic",
    "london", "berlin", "zurich", "stockholm", "brussels", "amsterdam",
    "liechtenstein", "luxembourg", "estonia", "lithuania",
]

# Stage-fit keywords
EARLY_STAGE_KEYWORDS = [
    "seed", "series a", "pre-seed", "early stage", "startup", "founded",
    "launch", "raised", "funding round", "angel", "accelerator", "incubator",
]

# Source authority scores (default 0.5)
SOURCE_AUTHORITY: dict[str, float] = {
    "person": 0.6,
    "association": 0.7,
    "newsletter": 0.5,
    "university": 0.4,
    "conference": 0.5,
    "vc": 0.8,
    "regulator": 0.9,
}


def score_finding(raw_finding: dict, source: Source) -> dict:
    """Score a raw finding for relevance to Stratum's thesis.

    Uses LLM-provided relevance_score when available (primary),
    falls back to keyword-based scoring (secondary).
    Adds relevance_score and dedup_hash to the finding dict.
    """
    title = raw_finding.get("title", "")
    summary = raw_finding.get("summary", "")
    text = f"{title} {summary}".lower()

    # --- Auto-tag verticals (always runs, regardless of scoring method) ---
    vertical_scores = {}
    for vertical, keywords in VERTICAL_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw.lower() in text)
        vertical_scores[vertical] = min(hits / 3, 1.0)

    auto_tags = [v for v, s in vertical_scores.items() if s >= 0.3]
    existing_tags = raw_finding.get("vertical_tags", [])
    all_tags = list(set(existing_tags + auto_tags))

    # --- Score: prefer LLM score, fallback to keyword ---
    llm_score = raw_finding.get("relevance_score")

    if llm_score is not None and isinstance(llm_score, (int, float)):
        # LLM scored it — use directly, with light source authority adjustment
        authority = SOURCE_AUTHORITY.get(source.category, 0.5)
        # Blend: 85% LLM judgment + 15% source authority
        relevance_score = 0.85 * float(llm_score) + 0.15 * authority
    else:
        # Fallback: keyword-based scoring
        vertical_alignment = max(vertical_scores.values()) if vertical_scores else 0.0

        geo_hits = sum(1 for kw in EUROPE_KEYWORDS if kw.lower() in text)
        geographic_relevance = min(geo_hits / 2, 1.0)

        stage_hits = sum(1 for kw in EARLY_STAGE_KEYWORDS if kw.lower() in text)
        stage_fit = min(stage_hits / 2, 1.0)

        recency = 1.0
        authority = SOURCE_AUTHORITY.get(source.category, 0.5)

        CATEGORY_BONUS = {
            "regulatory": 0.15,
            "funding_round": 0.15,
            "product_launch": 0.10,
            "partnership": 0.05,
            "research": 0.05,
        }
        category_bonus = CATEGORY_BONUS.get(raw_finding.get("category", ""), 0.0)

        relevance_score = (
            settings.score_weight_vertical * vertical_alignment
            + settings.score_weight_geographic * geographic_relevance
            + settings.score_weight_stage * stage_fit
            + settings.score_weight_recency * recency
            + settings.score_weight_authority * authority
            + category_bonus
        )

    # Clamp to [0, 1]
    relevance_score = max(0.0, min(1.0, relevance_score))

    # Compute dedup hash
    date_bucket = date.today().isoformat()
    dedup_input = f"{title.lower().strip()}|{source.id}|{date_bucket}"
    dedup_hash = hashlib.sha256(dedup_input.encode()).hexdigest()

    return {
        **raw_finding,
        "relevance_score": round(relevance_score, 4),
        "vertical_tags": all_tags,
        "dedup_hash": dedup_hash,
    }
