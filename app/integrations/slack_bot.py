"""Slack integration: morning digest (push) and Q&A (pull) via Slack Bolt."""

import hashlib
from datetime import datetime, timedelta, timezone

import structlog
from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.database import async_session_factory
from app.integrations.llm import call_llm
from app.models import Evidence, Finding, Notification

logger = structlog.get_logger()

# Initialize Slack Bolt app (lazily -- only if tokens are configured)
_slack_app: AsyncApp | None = None


def get_slack_app() -> AsyncApp | None:
    global _slack_app
    if _slack_app is not None:
        return _slack_app

    if not settings.slack_bot_token or not settings.slack_signing_secret:
        logger.warning("slack_not_configured")
        return None

    _slack_app = AsyncApp(
        token=settings.slack_bot_token,
        signing_secret=settings.slack_signing_secret,
    )

    # Register event handlers
    @_slack_app.event("app_mention")
    async def handle_mention(event, say):
        await _handle_question(event, say)

    @_slack_app.command("/lexie")
    async def handle_slash(ack, command, say):
        await ack()
        await _handle_question(
            {"text": command["text"], "ts": command.get("trigger_id", "")},
            say,
        )

    return _slack_app


async def _handle_question(event: dict, say) -> None:
    """Handle a Slack Q&A question: search findings, generate grounded answer."""
    question = event.get("text", "").strip()
    # Remove bot mention if present
    if "<@" in question:
        question = question.split(">", 1)[-1].strip()

    if not question:
        await say("Please ask a question and I'll search our sourcing database for answers.")
        return

    thread_ts = event.get("ts")

    # Search findings
    async with async_session_factory() as db:
        findings = await _search_findings(db, question)

    if not findings:
        await say(
            text="I couldn't find any relevant findings in our database for that question.",
            thread_ts=thread_ts,
        )
        return

    # Format evidence for LLM
    evidence_text = _format_evidence_for_llm(findings)

    # Generate grounded answer
    answer = await _generate_grounded_answer(question, evidence_text)

    # Post answer
    await say(text=answer, thread_ts=thread_ts)


async def _search_findings(db: AsyncSession, query: str, limit: int = 10) -> list[Finding]:
    """Search findings by text relevance."""
    words = query.lower().split()
    stmt = (
        select(Finding)
        .options(selectinload(Finding.evidence_items))
        .where(Finding.status != "dismissed")
        .order_by(Finding.relevance_score.desc())
        .limit(50)
    )
    result = await db.execute(stmt)
    all_findings = result.scalars().all()

    # Simple keyword ranking
    scored = []
    for f in all_findings:
        text = f"{f.title} {f.summary}".lower()
        hits = sum(1 for w in words if w in text)
        if hits > 0:
            scored.append((hits, f))

    scored.sort(key=lambda x: (-x[0], -x[1].relevance_score))
    return [f for _, f in scored[:limit]]


def _format_evidence_for_llm(findings: list[Finding]) -> str:
    """Format findings and their evidence for LLM context."""
    parts = []
    for i, f in enumerate(findings, 1):
        evidence_lines = []
        for ev in f.evidence_items[:3]:
            evidence_lines.append(f"  - URL: {ev.url}\n    Excerpt: {ev.excerpt}")

        evidence_block = "\n".join(evidence_lines) if evidence_lines else "  (no evidence links)"
        parts.append(
            f"[{i}] {f.title}\n"
            f"Score: {f.relevance_score:.2f} | Category: {f.category} | "
            f"Date: {f.created_at.strftime('%Y-%m-%d')}\n"
            f"Summary: {f.summary}\n"
            f"Evidence:\n{evidence_block}"
        )
    return "\n\n".join(parts)


async def _generate_grounded_answer(question: str, evidence: str) -> str:
    """Generate a Slack-friendly answer grounded in evidence."""
    prompt = (
        "You are Lexie, the sourcing analyst for Stratum 3Ventures. "
        "Answer the user's question using ONLY the evidence below. "
        "Cite sources with [1], [2], etc. "
        "If the evidence doesn't cover the question, say so honestly. "
        "Keep your answer concise and Slack-friendly (no markdown headers, use bullet points).\n\n"
        f"Evidence:\n{evidence}\n\n"
        f"Question: {question}"
    )

    return await call_llm(prompt=prompt, max_tokens=1500, temperature=0.3)


# --- Morning Digest ---


async def send_morning_digest() -> bool:
    """Build and send the morning digest to Slack.

    Returns:
        True if digest was sent successfully.
    """
    if not settings.slack_bot_token:
        logger.warning("slack_not_configured_for_digest")
        return False

    client = AsyncWebClient(token=settings.slack_bot_token)

    async with async_session_factory() as db:
        # Get findings from last 24 hours
        since = datetime.now(timezone.utc) - timedelta(hours=24)
        stmt = (
            select(Finding)
            .options(selectinload(Finding.evidence_items))
            .where(Finding.created_at >= since, Finding.status != "dismissed")
            .order_by(Finding.relevance_score.desc())
            .limit(20)
        )
        result = await db.execute(stmt)
        findings = result.scalars().all()

        if not findings:
            logger.info("no_findings_for_digest")
            return False

        top_findings = findings[:5]

        # Collect vertical tags across all findings as radar topics
        from collections import Counter

        # Canonical vertical names
        TAG_CANONICAL = {
            "identity_permissioning": "Identity & Permissioning",
            "wallets_key_management": "Wallets & Key Management",
            "compliance_trust": "Compliance & Trust Infrastructure",
            "data_oracles_middleware": "Data, Oracles & Middleware",
        }
        tag_counts: Counter[str] = Counter()
        for f in findings:
            if f.vertical_tags:
                for tag in f.vertical_tags:
                    key = tag.lower().replace(" ", "_").replace(",", "").replace("&", "")
                    canonical = TAG_CANONICAL.get(key, tag)
                    tag_counts[canonical] += 1
        top_entities = [t for t, _ in tag_counts.most_common(6)]

        # Build Slack Block Kit message
        blocks = _build_digest_blocks(top_findings, top_entities)

        # Check for duplicate notification
        digest_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        payload_hash = hashlib.sha256(f"digest:{digest_date}".encode()).hexdigest()

        existing = await db.execute(
            select(Notification).where(
                Notification.channel == "slack_digest",
                Notification.payload_hash == payload_hash,
            )
        )
        if existing.scalar_one_or_none():
            logger.info("digest_already_sent", date=digest_date)
            return False

        # Send to Slack
        response = await client.chat_postMessage(
            channel=settings.slack_channel_sourcing,
            text=f"Lexie's Morning Brief -- {digest_date}",
            blocks=blocks,
        )

        # Record notification
        notification = Notification(
            channel="slack_digest",
            channel_ref=response.get("ts"),
            payload_hash=payload_hash,
        )
        db.add(notification)
        await db.commit()

    logger.info("digest_sent", findings_count=len(top_findings))
    return True


def _build_digest_blocks(findings: list[Finding], entities: list[str]) -> list[dict]:
    """Build Slack Block Kit blocks for the morning digest."""
    today = datetime.now(timezone.utc).strftime("%A, %B %d, %Y")

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"Lexie's Morning Brief -- {today}"},
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Top {len(findings)} Findings*",
            },
        },
    ]

    SCORE_EMOJIS = {5: "🔴", 4: "🟠", 3: "🟡", 2: "🔵", 1: "⚪"}
    CATEGORY_LABELS = {
        "funding_round": "💰 Funding",
        "product_launch": "🚀 Launch",
        "partnership": "🤝 Partnership",
        "regulatory": "⚖️ Regulatory",
        "hiring": "👤 Hiring",
        "research": "📊 Research",
        "market_move": "📈 Market",
        "opinion": "💬 Opinion",
    }

    for i, f in enumerate(findings, 1):
        # 1-5 dot scale
        dots = min(5, max(1, round(f.relevance_score * 5)))
        dot_indicator = SCORE_EMOJIS.get(dots, "⚪") * dots
        cat_label = CATEGORY_LABELS.get(f.category, f.category or "general")

        evidence_links = ""
        if f.evidence_items:
            links = [f"<{ev.url}|source>" for ev in f.evidence_items[:2]]
            evidence_links = " | ".join(links)

        # Truncate summary at word boundary
        summary = f.summary
        if len(summary) > 200:
            summary = summary[:200].rsplit(" ", 1)[0] + "…"

        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*{i}. {f.title}*\n"
                    f"_{summary}_\n"
                    f"{dot_indicator} {cat_label}"
                    + (f" | {evidence_links}" if evidence_links else "")
                ),
            },
        })

    if entities:
        blocks.extend([
            {"type": "divider"},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Topics on the Radar:* {', '.join(entities)}",
                },
            },
        ])

    blocks.append({
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": "Powered by Stratum 3V Sourcing Monitor | Ask me anything with @Lexie",
            }
        ],
    })

    return blocks
