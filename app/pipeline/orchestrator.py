"""Main scan orchestrator: iterates sources, fetches content, produces findings.

Follows the ncf-dataroom multi-stage pipeline pattern with per-source error isolation,
bounded concurrency, and stage-level tracking for observability.
"""

import asyncio
import time
from datetime import datetime, timezone

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import async_session_factory
from app.models import ScanRun, Snapshot, Source
from app.pipeline.analyzer import analyze_diff
from app.pipeline.deduplicator import clear_embedding_cache, is_duplicate, is_semantic_duplicate
from app.pipeline.differ import compute_diff
from app.pipeline.scorer import score_finding
from app.pipeline.tracker import PipelineTracker, ScanStage
from app.sources.registry import close_all_fetchers, get_fetcher

logger = structlog.get_logger()


async def run_scan() -> int:
    """Execute a full scan of all active sources.

    Returns:
        The scan_run ID.
    """
    async with async_session_factory() as db:
        # Create scan run record
        run = ScanRun(started_at=datetime.now(timezone.utc))
        db.add(run)
        await db.flush()
        run_id = run.id

        # Load active sources
        result = await db.execute(
            select(Source).where(Source.is_active.is_(True))
        )
        sources = result.scalars().all()
        run.sources_total = len(sources)
        await db.commit()

    logger.info("scan_started", run_id=run_id, sources=len(sources))

    # Process sources with bounded concurrency
    semaphore = asyncio.Semaphore(settings.fetch_concurrency)
    errors: list[dict] = []
    ok_count = 0
    findings_count = 0

    async def process_source(source: Source) -> None:
        nonlocal ok_count, findings_count
        async with semaphore:
            try:
                count = await _scan_single_source(source, run_id)
                ok_count += 1
                findings_count += count
            except Exception as e:
                logger.error(
                    "source_scan_failed",
                    source_id=source.id,
                    source_name=source.name,
                    error=str(e),
                )
                errors.append({
                    "source_id": source.id,
                    "source_name": source.name,
                    "error": str(e),
                })

    # Run all sources concurrently (bounded by semaphore)
    await asyncio.gather(
        *(process_source(s) for s in sources),
        return_exceptions=True,
    )

    # Clean up resources
    await close_all_fetchers()
    clear_embedding_cache()

    # Update scan run with results
    async with async_session_factory() as db:
        run = await db.get(ScanRun, run_id)
        run.finished_at = datetime.now(timezone.utc)
        run.status = "completed_with_errors" if errors else "completed"
        run.sources_ok = ok_count
        run.sources_failed = len(errors)
        run.findings_count = findings_count
        run.error_log = errors
        await db.commit()

    logger.info(
        "scan_completed",
        run_id=run_id,
        ok=ok_count,
        failed=len(errors),
        findings=findings_count,
    )
    return run_id


async def _scan_single_source(source: Source, run_id: int) -> int:
    """Scan a single source: fetch → diff → analyze → score → dedup → store.

    Each stage is tracked via PipelineTracker for observability (ncf pattern).

    Returns:
        Number of new findings created.
    """
    tracker = PipelineTracker(run_id=run_id, source_id=source.id, source_name=source.name)
    fetcher = get_fetcher(source.fetch_strategy)

    if not source.url:
        logger.info("source_no_url", source_id=source.id, name=source.name)
        return 0

    # --- Stage: FETCH ---
    tracker.start(ScanStage.FETCH)
    t0 = time.monotonic()
    result = await fetcher.fetch(source.url, source.config)
    fetch_ms = int((time.monotonic() - t0) * 1000)

    async with async_session_factory() as db:
        # Store snapshot regardless of success/failure
        snapshot = Snapshot(
            source_id=source.id,
            run_id=run_id,
            content_hash=result.content_hash or "",
            raw_content=result.content if not result.error else None,
            fetched_at=result.fetched_at,
            fetch_duration_ms=result.duration_ms,
            error=result.error,
        )
        db.add(snapshot)
        await db.flush()

        if result.error:
            tracker.fail(ScanStage.FETCH, error=result.error)
            await db.commit()
            return 0

        tracker.complete(ScanStage.FETCH, duration_ms=fetch_ms)

        # --- Stage: DIFF ---
        tracker.start(ScanStage.DIFF)
        t0 = time.monotonic()
        diff_text = await compute_diff(db, source.id, result.content_hash, result.content, run_id=run_id)
        diff_ms = int((time.monotonic() - t0) * 1000)

        if not diff_text:
            tracker.complete(ScanStage.DIFF, duration_ms=diff_ms, details={"changed": False})
            await db.commit()
            return 0

        tracker.complete(ScanStage.DIFF, duration_ms=diff_ms, details={"changed": True})

        # --- Stage: ANALYZE (LLM, per-category prompt) ---
        tracker.start(ScanStage.ANALYZE)
        t0 = time.monotonic()
        raw_findings = await analyze_diff(
            diff_text=diff_text,
            source_name=source.name,
            source_category=source.category,
            source_url=source.url,
        )
        analyze_ms = int((time.monotonic() - t0) * 1000)
        tracker.complete(
            ScanStage.ANALYZE,
            duration_ms=analyze_ms,
            details={"raw_findings": len(raw_findings)},
        )

        # --- Stage: SCORE + DEDUP + STORE ---
        tracker.start(ScanStage.STORE)
        t0 = time.monotonic()
        new_count = 0
        dupes = 0

        for raw in raw_findings:
            # Score (applies vertical alignment, geographic, stage, authority weights)
            scored = score_finding(raw, source)

            # Dedup layer 1: exact hash (same title + source + date)
            if await is_duplicate(db, scored["dedup_hash"]):
                dupes += 1
                continue

            # Dedup layer 2: semantic similarity (same news from different sources)
            if await is_semantic_duplicate(
                db, scored["title"], scored["summary"], source.id
            ):
                dupes += 1
                continue

            from app.models import Evidence, Finding

            finding = Finding(
                run_id=run_id,
                source_id=source.id,
                title=scored["title"],
                summary=scored["summary"],
                category=scored.get("category"),
                relevance_score=scored["relevance_score"],
                vertical_tags=scored.get("vertical_tags", []),
                dedup_hash=scored["dedup_hash"],
            )
            db.add(finding)
            await db.flush()

            # Store evidence items
            for ev in scored.get("evidence", []):
                evidence = Evidence(
                    finding_id=finding.id,
                    url=ev.get("url", source.url),
                    excerpt=ev.get("excerpt", ""),
                    captured_at=result.fetched_at,
                )
                db.add(evidence)

            new_count += 1

        await db.commit()
        store_ms = int((time.monotonic() - t0) * 1000)
        tracker.complete(
            ScanStage.STORE,
            duration_ms=store_ms,
            details={"new": new_count, "duplicates": dupes},
        )

    logger.info(
        "source_scanned",
        source_id=source.id,
        name=source.name,
        new_findings=new_count,
        duplicates=dupes,
        stages=tracker.to_dict()["stages"],
    )
    return new_count
