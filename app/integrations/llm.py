"""LLM client abstraction with four provider tiers.

Priority order (matches ncf-dataroom production pattern):
1. OpenClaw gateway (production on Railway -- WebSocket RPC, handles Codex OAuth internally)
2. Codex Responses API (local dev fallback -- direct streaming via OAuth minter)
3. Anthropic API (direct, if ANTHROPIC_API_KEY set)
4. OpenAI API (direct, if OPENAI_API_KEY set)
"""

import asyncio
import json
import time
import uuid

import httpx
import structlog

from app.config import settings

logger = structlog.get_logger()

# --- Token cache for OAuth minter ---
_token_cache: dict = {"token": None, "expires_at": 0}


async def call_llm(
    prompt: str,
    system: str = "",
    max_tokens: int = 4096,
    temperature: float = 0.2,
    model: str | None = None,
) -> str:
    """Route LLM call to the best available provider."""
    model = model or settings.llm_model

    # 1. OpenClaw gateway (production path -- same as ncf-dataroom)
    if settings.openclaw_gateway_url or settings.openclaw_internal_port:
        try:
            return await _call_openclaw(prompt, system, max_tokens, temperature, model)
        except Exception as e:
            logger.warning("openclaw_unavailable", error=str(e))

    # 2. Codex Responses API (local dev fallback -- direct streaming via OAuth minter)
    if settings.oauth_minter_url and settings.oauth_minter_key:
        try:
            return await _call_codex(prompt, system, max_tokens, temperature, model)
        except Exception as e:
            logger.warning("codex_unavailable", error=str(e))

    # 3. Anthropic API (direct)
    if settings.anthropic_api_key:
        return await _call_anthropic(prompt, system, max_tokens, temperature, model)

    # 4. OpenAI API (direct, with API key)
    if settings.openai_api_key:
        return await _call_openai_direct(prompt, system, max_tokens, temperature, model)

    raise RuntimeError(
        "No LLM provider available. Configure one of: "
        "OAUTH_MINTER_URL+KEY (Codex), OPENCLAW_GATEWAY_URL, "
        "ANTHROPIC_API_KEY, or OPENAI_API_KEY."
    )


# --- Codex Responses API (chatgpt.com/backend-api/codex/responses) ---


async def _mint_token() -> str:
    """Get or refresh OAuth token from minter. Cached until 5 min before expiry."""
    if _token_cache["token"] and time.time() < _token_cache["expires_at"] - 300:
        return _token_cache["token"]

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{settings.oauth_minter_url}/mint",
            headers={"Authorization": f"Bearer {settings.oauth_minter_key}"},
        )
        resp.raise_for_status()
        data = resp.json()

    _token_cache["token"] = data["access_token"]
    _token_cache["expires_at"] = time.time() + data.get("expires_in", 3600)
    logger.debug("oauth_token_minted", expires_in=data.get("expires_in"))
    return _token_cache["token"]


async def _call_codex(
    prompt: str, system: str, max_tokens: int, temperature: float, model: str
) -> str:
    """Call LLM via Codex Responses API (streaming, ChatGPT Pro)."""
    token = await _mint_token()

    # Codex responses endpoint requires: model, instructions, input (list), stream=True, store=False
    codex_model = model if "codex" in model or model.startswith("gpt-5") else "gpt-5.4"

    payload = {
        "model": codex_model,
        "instructions": system or "You are a helpful assistant.",
        "input": [{"role": "user", "content": prompt}],
        "stream": True,
        "store": False,
    }

    collected = []
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream(
            "POST",
            "https://chatgpt.com/backend-api/codex/responses",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise RuntimeError(f"Codex API {resp.status_code}: {body.decode()[:300]}")
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    evt = json.loads(data_str)
                    if evt.get("type") == "response.output_text.delta":
                        collected.append(evt.get("delta", ""))
                except json.JSONDecodeError:
                    pass

    text = "".join(collected).strip()
    logger.debug("llm_call", provider="codex", model=codex_model, chars=len(text))
    return text


# --- OpenClaw Gateway (WebSocket RPC, ncf-dataroom pattern) ---


async def _call_openclaw(
    prompt: str, system: str, max_tokens: int, temperature: float, model: str
) -> str:
    """Call LLM via OpenClaw gateway using WebSocket RPC."""
    import websockets

    ws_url = settings.openclaw_gateway_url or f"ws://127.0.0.1:{settings.openclaw_internal_port}"
    gateway_token = settings.openclaw_gateway_token

    session_key = f"pipeline-{uuid.uuid4().hex[:12]}"
    idempotency_key = f"llm-{uuid.uuid4().hex[:12]}"
    message = f"[System: {system}]\n\n{prompt}" if system else prompt

    async with websockets.connect(
        ws_url,
        additional_headers={"Authorization": f"Bearer {gateway_token}"} if gateway_token else {},
        open_timeout=20.0,
        close_timeout=5.0,
    ) as ws:
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=20.0)
            frame = json.loads(raw)
            if frame.get("type") == "event" and frame.get("event") == "connect.challenge":
                break

        connect_id = f"connect-{uuid.uuid4().hex[:8]}"
        connect_payload = {
            "minProtocol": 3,
            "maxProtocol": 3,
            "client": {"id": "stratum-pipeline", "version": "1.0", "platform": "python"},
            "caps": [],
            "commands": [],
            "role": "operator",
            "scopes": ["operator.admin"],
        }
        if gateway_token:
            connect_payload["auth"] = {"token": gateway_token}

        await ws.send(json.dumps({
            "type": "req", "id": connect_id, "method": "connect", "params": connect_payload,
        }))

        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=20.0)
            frame = json.loads(raw)
            if frame.get("type") == "res" and frame.get("id") == connect_id:
                if not frame.get("ok"):
                    raise RuntimeError(f"OpenClaw connect failed: {frame.get('error')}")
                break

        send_id = str(uuid.uuid4())
        await ws.send(json.dumps({
            "type": "req", "id": send_id, "method": "chat.send",
            "params": {"sessionKey": session_key, "message": message, "idempotencyKey": idempotency_key},
        }))

        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=20.0)
            frame = json.loads(raw)
            if frame.get("type") == "res" and frame.get("id") == send_id:
                break

        collected_text = []
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=300.0)
            frame = json.loads(raw)
            if frame.get("type") != "event":
                continue
            event_name = frame.get("event", "")
            payload = frame.get("payload", {})
            if event_name == "chat":
                for item in payload.get("payloads", []):
                    if isinstance(item, dict) and item.get("text") and not item.get("isError"):
                        collected_text.append(item["text"])
                state = payload.get("state", "")
                if state in ("final", "error"):
                    if state == "error":
                        error_texts = [
                            item.get("text", "") for item in payload.get("payloads", []) if item.get("isError")
                        ]
                        raise RuntimeError(f"OpenClaw error: {' '.join(error_texts)}")
                    break

    text = "".join(collected_text).strip()
    logger.debug("llm_call", provider="openclaw", model=model, chars=len(text))
    return text


# --- Direct Anthropic API ---


async def _call_anthropic(
    prompt: str, system: str, max_tokens: int, temperature: float, model: str
) -> str:
    """Call Anthropic Claude API directly."""
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text
    logger.debug(
        "llm_call", provider="anthropic", model=model,
        input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens,
    )
    return text


# --- Direct OpenAI API (with API key, not OAuth) ---


async def _call_openai_direct(
    prompt: str, system: str, max_tokens: int, temperature: float, model: str
) -> str:
    """Call OpenAI API with a standard API key."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature},
        )
        resp.raise_for_status()
        data = resp.json()

    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    logger.debug(
        "llm_call", provider="openai", model=model,
        input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
    )
    return text
