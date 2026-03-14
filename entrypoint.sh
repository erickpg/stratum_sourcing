#!/bin/bash
set -e

echo "=== Stratum Sourcing Monitor - Starting ==="

# --- Cron task mode: run a single task and exit ---
# Set TASK_MODE env var on Railway cron services to skip the web server.
if [ -n "${TASK_MODE:-}" ]; then
    echo "--- Task mode: $TASK_MODE ---"
    python -m alembic upgrade head
    case "$TASK_MODE" in
        nightly-scan)
            python -m app.tasks.nightly_scan
            ;;
        morning-digest)
            python -m app.tasks.morning_digest
            ;;
        notion-export)
            python -m app.tasks.notion_export
            ;;
        *)
            echo "Unknown TASK_MODE: $TASK_MODE"
            exit 1
            ;;
    esac
    echo "--- Task $TASK_MODE complete ---"
    exit 0
fi

# --- Web server mode (default) ---

# Ensure persistent volume directories exist
mkdir -p "${DATA_DIR:-/data}/browser/profile"
mkdir -p "${DATA_DIR:-/data}/cache"
mkdir -p "${DATA_DIR:-/data}/logs"

echo "--- Running database migrations ---"
python -m alembic upgrade head

echo "--- Seeding sources (if needed) ---"
python -c "
import asyncio, json, sys
from pathlib import Path

async def seed():
    from app.database import async_session_factory
    from app.models import Source
    from sqlalchemy import select

    async with async_session_factory() as db:
        result = await db.execute(select(Source).limit(1))
        if result.scalar_one_or_none():
            print('Sources already seeded, skipping')
            return

        seeds_file = Path('seeds/sources.json')
        if not seeds_file.exists():
            print('No seeds/sources.json found, skipping')
            return

        sources = json.loads(seeds_file.read_text())
        for s in sources:
            db.add(Source(
                name=s['name'],
                category=s['category'],
                fetch_strategy=s['fetch_strategy'],
                url=s.get('url'),
                secondary_urls=s.get('secondary_urls', []),
                config=s.get('config', {}),
                verticals=s.get('verticals', []),
                description=s.get('description'),
                is_active=s.get('is_active', True),
            ))
        await db.commit()
        print(f'Seeded {len(sources)} sources')

asyncio.run(seed())
" || echo "Warning: Source seeding failed, continuing..."

FASTAPI_PORT="${PORT:-8081}"
echo "--- Starting FastAPI sidecar (port $FASTAPI_PORT) ---"
uvicorn app.main:app --host 0.0.0.0 --port "$FASTAPI_PORT" --log-level info &
FASTAPI_PID=$!

# Start OpenClaw gateway if configured
if [ -f "src/server.js" ] && [ -n "${OPENCLAW_STATE_DIR:-}" ]; then
    echo "--- Starting OpenClaw gateway (port 8080) ---"

    # If OAuth minter is configured, authenticate OpenClaw with Codex
    if [ -n "${OAUTH_MINTER_URL:-}" ] && [ -n "${OAUTH_MINTER_KEY:-}" ]; then
        echo "--- Configuring OpenClaw Codex OAuth via minter ---"
        # OpenClaw reads auth profiles from its state dir
        export OPENCLAW_CODEX_PROVIDER="${OPENCLAW_CODEX_PROVIDER:-openai-codex}"
    fi

    cd src && node server.js &
    OPENCLAW_PID=$!
    cd ..
else
    echo "--- OpenClaw gateway not configured, skipping ---"
    OPENCLAW_PID=""
fi

# Wait for either process to exit
cleanup() {
    echo "=== Shutting down ==="
    kill $FASTAPI_PID 2>/dev/null || true
    [ -n "$OPENCLAW_PID" ] && kill $OPENCLAW_PID 2>/dev/null || true
    wait
}
trap cleanup SIGTERM SIGINT

if [ -n "$OPENCLAW_PID" ]; then
    wait -n $FASTAPI_PID $OPENCLAW_PID
else
    wait $FASTAPI_PID
fi
