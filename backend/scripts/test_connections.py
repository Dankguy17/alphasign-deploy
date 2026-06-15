"""
test_connections.py

Standalone sanity check for all three external services AlphaSign depends on:

  1. Featherless API   — open-source model inference (OpenAI-compatible)
  2. AI/ML API         — hosted model catalog (OpenAI-compatible)
  3. Band              — multi-agent chat-room platform (band-sdk)

Run this BEFORE writing any agent logic. It does nothing with Band rooms
or message routing yet — it just confirms each set of credentials in
.env / agent_config.yaml actually authenticates.

Usage:
    uv run python scripts/test_connections.py

Requires (see backend/pyproject.toml or requirements.txt):
    openai
    python-dotenv
    band-sdk (any extra, e.g. band-sdk[anthropic] or band-sdk[pydantic-ai])
    pyyaml
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────
# 1. Featherless API
# ─────────────────────────────────────────────────────────────
def test_featherless() -> bool:
    print("\n[1/3] Testing Featherless API...")
    try:
        from openai import OpenAI

        api_key = os.getenv("FEATHERLESS_API_KEY")
        base_url = os.getenv("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1")

        if not api_key or api_key.startswith("your_"):
            print("  SKIP: FEATHERLESS_API_KEY not set in .env")
            return False

        client = OpenAI(api_key=api_key, base_url=base_url)

        # Small, cheap open-source model for a quick smoke test.
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[
                {"role": "user", "content": "Reply with exactly one word: OK"}
            ],
            max_tokens=10,
        )
        text = response.choices[0].message.content.strip()
        print(f"  OK — model responded: {text!r}")
        return True

    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# 2. AI/ML API
# ─────────────────────────────────────────────────────────────
def test_aimlapi() -> bool:
    print("\n[2/3] Testing AI/ML API...")
    try:
        from openai import OpenAI

        api_key = os.getenv("AIML_API_KEY")
        base_url = os.getenv("AIML_BASE_URL", "https://api.aimlapi.com/v1")

        if not api_key or api_key.startswith("your_"):
            print("  SKIP: AIML_API_KEY not set in .env")
            return False

        client = OpenAI(api_key=api_key, base_url=base_url)

        # gpt-4o-mini-style cheap model — adjust to whatever's cheapest
        # on your AI/ML API plan if this model isn't available.
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Reply with exactly one word: OK"}
            ],
            max_tokens=10,
        )
        text = response.choices[0].message.content.strip()
        print(f"  OK — model responded: {text!r}")
        return True

    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# 3. Band — connection + identity check only
#
# This does NOT start the long-running agent loop (agent.run()).
# It just validates that the agent_id/api_key pair in
# agent_config.yaml authenticates against the Band platform via
# the Request API (GET /agent/me).
# ─────────────────────────────────────────────────────────────
def test_band() -> bool:
    print("\n[3/3] Testing Band connection...")
    try:
        import httpx
        from thenvoi.config import load_agent_config
    except ImportError as e:
        print(f"  FAIL: missing dependency ({e}).")
        print("  Run: uv add 'band-sdk[anthropic]' httpx pyyaml")
        return False

    try:
        # Tests the "executive" agent's credentials. Change the name to
        # test a different agent defined in agent_config.yaml.
        agent_id, api_key = load_agent_config("executive")
    except Exception as e:
        print(f"  SKIP: could not load agent_config.yaml ({e})")
        print("  Copy agent_config.yaml.example -> agent_config.yaml and fill in real values.")
        return False

    rest_url = os.getenv("THENVOI_REST_URL", "https://app.band.ai")

    try:
        response = httpx.get(
            f"{rest_url}/api/v1/agent/me",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()
        print(f"  OK — authenticated as agent: {data.get('name', agent_id)}")
        return True

    except httpx.HTTPStatusError as e:
        print(f"  FAIL: HTTP {e.response.status_code} — check agent_id/api_key in agent_config.yaml")
        return False
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("AlphaSign — API Connection Test")
    print("=" * 60)

    results = {
        "Featherless": test_featherless(),
        "AI/ML API": test_aimlapi(),
        "Band": test_band(),
    }

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL/SKIP"
        print(f"  {name:<15} {status}")

    if not all(results.values()):
        print("\nOne or more services did not connect. Fix the issues above")
        print("before starting agent development.")
        sys.exit(1)
    else:
        print("\nAll services reachable. Ready to start building agents.")


if __name__ == "__main__":
    main()
