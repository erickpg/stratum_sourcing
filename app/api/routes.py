"""FastAPI routes: health, sources CRUD, task triggers."""

import asyncio

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.deps import get_session, verify_cron_secret
from app.models import Finding, Source
from app.schemas import (
    FindingResponse,
    HealthResponse,
    SourceCreate,
    SourceResponse,
    TaskTriggerResponse,
)

logger = structlog.get_logger()
router = APIRouter()


# --- Health ---


@router.get("/healthz", response_model=HealthResponse)
async def healthz():
    return HealthResponse(status="ok")


# --- Sources ---


@router.get("/api/sources", response_model=list[SourceResponse])
async def list_sources(
    category: str | None = None,
    active_only: bool = True,
    db: AsyncSession = Depends(get_session),
):
    stmt = select(Source)
    if active_only:
        stmt = stmt.where(Source.is_active.is_(True))
    if category:
        stmt = stmt.where(Source.category == category)
    stmt = stmt.order_by(Source.category, Source.name)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.post("/api/sources", response_model=SourceResponse, status_code=201)
async def create_source(
    source: SourceCreate,
    db: AsyncSession = Depends(get_session),
):
    db_source = Source(**source.model_dump())
    db.add(db_source)
    await db.flush()
    await db.refresh(db_source)
    return db_source


# --- Findings ---


@router.get("/api/findings", response_model=list[FindingResponse])
async def list_findings(
    limit: int = 20,
    status: str | None = None,
    db: AsyncSession = Depends(get_session),
):
    stmt = select(Finding).options(selectinload(Finding.evidence_items)).order_by(Finding.relevance_score.desc()).limit(limit)
    if status:
        stmt = stmt.where(Finding.status == status)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/api/findings/search")
async def search_findings(
    q: str,
    limit: int = 10,
    db: AsyncSession = Depends(get_session),
):
    """Search findings by text (used by Lexie Q&A)."""
    from sqlalchemy import and_

    # Split query into words so "JPMorgan tokenized" matches findings containing both
    words = q.strip().split()
    word_filters = [
        Finding.title.ilike(f"%{w}%") | Finding.summary.ilike(f"%{w}%")
        for w in words
        if w
    ]
    stmt = (
        select(Finding)
        .options(selectinload(Finding.evidence_items))
        .where(and_(*word_filters) if word_filters else Finding.id > 0)
        .order_by(Finding.relevance_score.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    findings = result.scalars().all()
    return [FindingResponse.model_validate(f) for f in findings]


# --- Task triggers (called by Railway cron or manual) ---


@router.post("/tasks/nightly-scan", response_model=TaskTriggerResponse)
async def trigger_nightly_scan(
    _secret: str = Depends(verify_cron_secret),
):
    from app.tasks.nightly_scan import run_nightly_scan

    asyncio.create_task(run_nightly_scan())
    return TaskTriggerResponse(status="accepted", message="Nightly scan started")


@router.post("/tasks/morning-digest", response_model=TaskTriggerResponse)
async def trigger_morning_digest(
    _secret: str = Depends(verify_cron_secret),
):
    from app.tasks.morning_digest import run_morning_digest

    asyncio.create_task(run_morning_digest())
    return TaskTriggerResponse(status="accepted", message="Morning digest started")


@router.post("/tasks/notion-sync", response_model=TaskTriggerResponse)
async def trigger_notion_sync(
    _secret: str = Depends(verify_cron_secret),
):
    from app.tasks.notion_export import run_notion_sync

    asyncio.create_task(run_notion_sync())
    return TaskTriggerResponse(status="accepted", message="Notion sync started")


@router.post("/admin/reset-findings")
async def reset_findings(
    _secret: str = Depends(verify_cron_secret),
    db: AsyncSession = Depends(get_session),
):
    """Clear all findings, evidence, notifications, snapshots, and scan runs."""
    from sqlalchemy import text

    await db.execute(text("DELETE FROM notifications"))
    await db.execute(text("DELETE FROM evidence"))
    await db.execute(text("DELETE FROM findings"))
    await db.execute(text("DELETE FROM snapshots"))
    await db.execute(text("DELETE FROM scan_runs"))
    await db.commit()
    logger.info("admin_reset_findings")
    return {"status": "ok", "message": "All findings, snapshots, and scan runs cleared"}
