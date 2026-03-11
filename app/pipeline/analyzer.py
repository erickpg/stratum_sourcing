"""LLM-based analysis: extract structured findings from content diffs.

Uses per-source-category prompts to tailor extraction to the type of source,
following the ncf-dataroom pattern of content-type-specific processing paths
with structured Pydantic output schemas.
"""

import json

import structlog
from pydantic import BaseModel

from app.integrations.llm import call_llm

logger = structlog.get_logger()


# --- Structured output schema (Pydantic, like ncf ClassificationResponse) ---


class EvidenceItem(BaseModel):
    url: str = ""
    excerpt: str = ""


class RawFinding(BaseModel):
    title: str
    summary: str
    category: str = "opinion"
    relevance_score: float | None = None
    vertical_tags: list[str] = []
    evidence: list[EvidenceItem] = []
    entities: list[str] = []


# --- Base system prompt (shared context for all categories) ---

BASE_SYSTEM = """\
You are a structured data extraction assistant for Stratum 3Ventures, \
a VC fund investing in Layer 3 infrastructure at the TradFi-DeFi intersection.

The fund's four verticals:
1. Identity & Permissioning (digital identity, eID, eIDAS, verifiable credentials, access control)
2. Wallets & Key Management (institutional wallets, MPC, custody, HSM, account abstraction)
3. Compliance & Trust Infrastructure (KYC/KYB, AML, sanctions, regtech, MiCA, DLT Pilot Regime)
4. Data, Oracles & Middleware (oracles, data feeds, tokenisation rails, interoperability, settlement)

The fund targets Seed/Series A European companies building regulated tokenised market infrastructure.

Always respond with valid JSON: {"findings": [...]}.
Each finding has:
- title (max 100 chars)
- summary (2-3 sentences on why this matters for Stratum)
- category: funding_round, product_launch, partnership, regulatory, hiring, research, market_move, opinion
- relevance_score: 0.0-1.0 float rating how relevant this is to Stratum's thesis. Use this scale:
  0.9-1.0: Direct hit (EU Seed/A startup in our verticals, or major regulatory shift like MiCA)
  0.7-0.9: Strong signal (funding in adjacent infra, key partnership, regulatory consultation)
  0.5-0.7: Relevant context (institutional adoption signal, market structure shift, thought leadership)
  0.3-0.5: Weak signal (tangentially related, US-only, late-stage)
  0.0-0.3: Noise (not relevant enough to surface)
- vertical_tags: one or more of [identity_permissioning, wallets_key_management, compliance_trust, data_oracles_middleware]
- evidence: [{url, excerpt}] with source URL and supporting quote
- entities: company/org names mentioned
If nothing noteworthy, return {"findings": []}."""


# --- Per-category worker prompts (different instructions per source type) ---

CATEGORY_PROMPTS: dict[str, str] = {
    "person": """\
You are analysing a post or activity update from **{source_name}**, a thought leader \
in fintech/blockchain/DeFi.

**What to extract:**
- Opinions or takes on tokenisation, DeFi infrastructure, or regulatory developments
- Mentions of specific companies (especially Seed/Series A European startups)
- Signals about market direction relevant to Stratum's four verticals
- Announcements about the person's own company, investments, or board roles
- Insights on institutional adoption of blockchain/tokenisation

**What to skip:** generic motivational content, reposts without commentary, purely personal \
updates, unrelated topics.

Source URL: {source_url}

Content:
{content}""",

    "association": """\
You are analysing content from **{source_name}**, an industry association or major \
financial/tech company active in blockchain/tokenisation.

**What to extract:**
- Research reports, whitepapers, or data releases about tokenisation or digital assets
- Product launches or strategic announcements about blockchain infrastructure
- Partnerships or integrations relevant to regulated tokenised markets
- Market structure changes (new trading venues, custody solutions, settlement layers)
- Hiring patterns or leadership changes signalling strategic direction in relevant verticals
- Pilot programme results or production deployments of tokenisation technology

**What to skip:** generic corporate marketing, unrelated business lines, routine financial results \
without tokenisation relevance.

Source URL: {source_url}

Content:
{content}""",

    "newsletter": """\
You are analysing a newsletter issue from **{source_name}**, a publication covering \
fintech, crypto, blockchain, or tokenisation.

**What to extract:**
- New companies or projects mentioned (especially European, Seed/Series A stage)
- Funding rounds in relevant verticals (identity, wallets, compliance, data/oracles)
- Regulatory developments (MiCA, DLT Pilot, eIDAS, national frameworks)
- Infrastructure launches or partnerships for institutional tokenisation
- Trend analysis or market maps touching Stratum's thesis
- Notable hires or team moves in relevant companies

**Extract each distinct newsworthy item as a separate finding.**

**What to skip:** market price commentary, trading signals, meme coins, generic DeFi yield \
farming, NFT drops.

Source URL: {source_url}

Content:
{content}""",

    "university": """\
You are analysing academic or research content from **{source_name}**.

**What to extract:**
- Research papers on blockchain regulation, DeFi governance, or tokenisation infrastructure
- Faculty or lab announcements about digital identity, financial compliance, or oracle systems
- Industry-academia collaborations in Stratum's verticals
- Students/graduates founding companies in the tokenisation/DeFi infrastructure space
- Policy recommendations or regulatory analysis from academic experts
- Grants or funding for relevant research

**What to skip:** unrelated research, general computer science, cryptocurrency price modelling.

Source URL: {source_url}

Content:
{content}""",

    "conference": """\
You are analysing content from **{source_name}**, a fintech/blockchain/tokenisation \
conference or event.

**What to extract:**
- New speakers or panellists that signal emerging companies or regulatory shifts
- Announced themes or tracks relevant to tokenisation infrastructure
- Companies presenting or sponsoring that operate in Stratum's verticals
- Partnerships or product announcements made at the event
- Hackathon winners or startup pitch competition results in relevant areas
- Regulatory panels or policy discussions on digital assets

**What to skip:** generic event logistics, ticket sales, social events, unrelated tracks.

Source URL: {source_url}

Content:
{content}""",

    "vc": """\
You are analysing content from **{source_name}**, a venture capital firm active in \
Web3, fintech, or blockchain.

**What to extract:**
- New portfolio investments (especially in identity, wallets, compliance, or data infrastructure)
- Fund announcements, new fund launches, or thesis updates for tokenisation infrastructure
- Analysis or thought leadership on DeFi, tokenisation, or regulated crypto markets
- Co-investment signals with other VCs in Stratum's focus areas
- Portfolio company milestones (product launches, partnerships, funding rounds)
- Exit events (acquisitions, IPOs) for companies in relevant verticals

**What to skip:** general VC market commentary unrelated to tokenisation, consumer crypto plays.

Source URL: {source_url}

Content:
{content}""",

    "regulator": """\
You are analysing content from **{source_name}**, a regulatory body, policy organisation, \
or standards association.

**THIS IS HIGH-PRIORITY CONTENT. Regulatory findings should be flagged even if subtle.**

**What to extract:**
- New regulations, guidelines, or consultations on digital assets or tokenisation
- MiCA implementation updates, DLT Pilot Regime progress, eIDAS developments
- Enforcement actions or regulatory opinions on DeFi, stablecoins, or custody
- Standards proposals for tokenised securities, digital identity, or compliance
- Personnel changes at regulatory bodies signalling policy direction
- Cross-border regulatory coordination (ESMA, BaFin, FCA, FINMA, etc.)
- Sandbox applications or approvals for tokenisation-related companies
- Speeches or publications by key regulators on digital finance topics

**What to skip:** general financial regulation unrelated to digital assets or tokenisation.

Source URL: {source_url}

Content:
{content}""",
}

# Fallback for unknown categories
DEFAULT_PROMPT = """\
You are analysing content from **{source_name}** (category: {source_category}).

Extract any findings relevant to Stratum 3Ventures' thesis: Layer 3 infrastructure \
at the TradFi-DeFi intersection (identity, wallets, compliance, data/oracles), \
targeting Seed/Series A European companies in regulated tokenised markets.

Source URL: {source_url}

Content:
{content}"""


async def analyze_diff(
    diff_text: str,
    source_name: str,
    source_category: str,
    source_url: str,
) -> list[dict]:
    """Use LLM to extract structured findings from new/changed content.

    Uses per-source-category prompts for tailored extraction, following
    the ncf-dataroom pattern of content-type-specific processing.

    Returns:
        List of raw finding dicts (before scoring), validated via Pydantic.
    """
    # Truncate content for LLM context window
    max_content = 30000
    if len(diff_text) > max_content:
        diff_text = diff_text[:max_content] + "\n\n[Content truncated]"

    # Select category-specific prompt
    prompt_template = CATEGORY_PROMPTS.get(source_category, DEFAULT_PROMPT)
    prompt = prompt_template.format(
        source_name=source_name,
        source_category=source_category,
        source_url=source_url,
        content=diff_text,
    )

    try:
        response = await call_llm(
            prompt=prompt,
            system=BASE_SYSTEM,
            max_tokens=4096,
        )

        # Parse JSON response (handle markdown code blocks)
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        parsed = json.loads(text)

        # Handle both {"findings": [...]} and bare [...] formats
        if isinstance(parsed, dict) and "findings" in parsed:
            findings_raw = parsed["findings"]
        elif isinstance(parsed, list):
            findings_raw = parsed
        else:
            findings_raw = [parsed]

        # Validate each finding through Pydantic schema
        validated = []
        for f in findings_raw:
            try:
                # LLM sometimes returns evidence as a single dict instead of a list
                if isinstance(f.get("evidence"), dict):
                    f["evidence"] = [f["evidence"]]
                finding = RawFinding.model_validate(f)
                validated.append(finding.model_dump())
            except Exception as e:
                logger.warning(
                    "finding_validation_error",
                    error=str(e),
                    raw=str(f)[:200],
                )

        logger.info(
            "analysis_complete",
            source=source_name,
            category=source_category,
            findings_count=len(validated),
        )
        return validated

    except json.JSONDecodeError as e:
        logger.error("analysis_json_error", source=source_name, error=str(e))
        return []
    except Exception as e:
        logger.error("analysis_failed", source=source_name, error=str(e))
        return []
