"""Notion integration: sync findings to the Ocean database.

Two-way sync:
- Push: create Notion pages for new findings (unsynced, non-dismissed)
- Pull: read status changes from Notion back into DB (only recent, non-terminal findings)
"""

import hashlib
from datetime import datetime, timedelta, timezone

import structlog
from notion_client import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.database import async_session_factory
from app.models import Finding, Notification

logger = structlog.get_logger()

_notion: AsyncClient | None = None


def _get_notion() -> AsyncClient | None:
    global _notion
    if _notion is not None:
        return _notion
    if not settings.notion_api_key:
        return None
    _notion = AsyncClient(auth=settings.notion_api_key)
    return _notion


async def sync_findings_to_notion() -> int:
    """Sync unsynced findings to the Notion Ocean database.

    Returns:
        Number of pages created/updated.
    """
    notion = _get_notion()
    if not notion:
        logger.warning("notion_not_configured")
        return 0

    if not settings.notion_ocean_database_id:
        logger.warning("notion_database_id_not_set")
        return 0

    synced = 0

    async with async_session_factory() as db:
        # Get findings not yet synced to Notion (with source for name)
        stmt = (
            select(Finding)
            .options(selectinload(Finding.evidence_items), selectinload(Finding.source))
            .where(Finding.notion_page_id.is_(None), Finding.status != "dismissed")
            .order_by(Finding.relevance_score.desc())
            .limit(50)
        )
        result = await db.execute(stmt)
        findings = result.scalars().all()

        for finding in findings:
            try:
                page_id = await _create_notion_page(notion, finding)
                finding.notion_page_id = page_id

                # Record notification
                payload_hash = hashlib.sha256(
                    f"notion:{finding.id}:{page_id}".encode()
                ).hexdigest()
                notification = Notification(
                    finding_id=finding.id,
                    channel="notion",
                    channel_ref=page_id,
                    payload_hash=payload_hash,
                )
                db.add(notification)
                synced += 1

            except Exception as e:
                logger.error(
                    "notion_sync_failed",
                    finding_id=finding.id,
                    error=str(e),
                )

        await db.commit()

    logger.info("notion_sync_complete", synced=synced)
    return synced


async def _create_notion_page(notion: AsyncClient, finding: Finding) -> str:
    """Create a Notion page for a finding in the Ocean database."""
    # Build properties
    properties = {
        "title": {"title": [{"text": {"content": finding.title[:100]}}]},
        "Status": {"select": {"name": finding.status.capitalize()}},
        "Score": {"number": round(finding.relevance_score, 2)},
        "Category": {"select": {"name": finding.category or "general"}},
        "Summary": {
            "rich_text": [{"text": {"content": finding.summary[:2000]}}]
        },
        "Source": {
            "rich_text": [{"text": {"content": finding.source.name if finding.source else "Unknown"}}]
        },
    }

    # Add verticals as multi-select
    if finding.vertical_tags:
        properties["Verticals"] = {
            "multi_select": [{"name": v} for v in finding.vertical_tags]
        }

    # Add primary evidence URL
    if finding.evidence_items:
        properties["Evidence URL"] = {"url": finding.evidence_items[0].url}

    # Build page body with evidence details
    children = [
        {
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{"text": {"content": "Summary"}}],
            },
        },
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"text": {"content": finding.summary}}],
            },
        },
    ]

    # Add evidence blocks
    if finding.evidence_items:
        children.append({
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{"text": {"content": "Evidence"}}],
            },
        })

        for ev in finding.evidence_items:
            children.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [
                        {"text": {"content": f"{ev.excerpt[:300]}\n"}},
                        {
                            "text": {
                                "content": ev.url,
                                "link": {"url": ev.url},
                            },
                        },
                    ],
                },
            })

    response = await notion.pages.create(
        parent={"database_id": settings.notion_ocean_database_id},
        properties=properties,
        children=children,
    )

    return response["id"]


async def pull_status_updates() -> int:
    """Pull status changes from Notion back into the database.

    Returns:
        Number of findings updated.
    """
    notion = _get_notion()
    if not notion or not settings.notion_ocean_database_id:
        return 0

    updated = 0

    async with async_session_factory() as db:
        # Only check recent, non-terminal findings (skip archived/dismissed older than 30 days)
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        stmt = (
            select(Finding)
            .where(
                Finding.notion_page_id.isnot(None),
                Finding.status.notin_(["archived", "dismissed"]),
                Finding.created_at >= cutoff,
            )
        )
        result = await db.execute(stmt)
        findings = result.scalars().all()

        # Status mapping from Notion to our model
        status_map = {
            "New": "new",
            "Reviewed": "reviewed",
            "Actionable": "actionable",
            "Dismissed": "dismissed",
            "Archived": "archived",
        }

        for finding in findings:
            try:
                page = await notion.pages.retrieve(finding.notion_page_id)
                notion_status = (
                    page.get("properties", {})
                    .get("Status", {})
                    .get("select", {})
                    .get("name", "")
                )
                mapped = status_map.get(notion_status)
                if mapped and mapped != finding.status:
                    finding.status = mapped
                    updated += 1
                    logger.info(
                        "status_updated_from_notion",
                        finding_id=finding.id,
                        new_status=mapped,
                    )
            except Exception as e:
                logger.warning(
                    "notion_status_pull_failed",
                    finding_id=finding.id,
                    error=str(e),
                )

        await db.commit()

    return updated
