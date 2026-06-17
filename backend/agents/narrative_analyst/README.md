# Narrative Analyst Agent

This agent researches ticker news, builds a structured Narrative Radar, and asks the Signal Processing and Latent State agents targeted follow-up questions.

## Why this is unique

Instead of returning a plain news summary, the agent produces:

- bullish thesis
- bearish thesis
- catalysts
- risk flags
- missing evidence
- Signal Processing request
- Latent State request
- source reliability tiering

That gives judges a visible multi-agent reasoning loop: the News agent does research, forms hypotheses, and asks the quant agents to verify or reject them.

## Source Reliability Engine

Every article is tagged with a tier and confidence:

- Tier 1, 0.92-0.97: SEC/company/government sources and official press-release wires.
- Tier 2, 0.84: major publications and recognized financial media.
- Tier 3, 0.72: analyst, industry research, and market commentary sources.
- Tier 4, 0.52-0.58: aggregators, reposted feeds, blogs, or unknown sources.

The Narrative Radar includes both per-article reliability and aggregate source
quality, so Executive can prefer higher-confidence evidence.

## Free-first data path

1. `NEWS_API_KEY`, if available.
2. Yahoo Finance RSS, no key.
3. `yfinance.Ticker.news`, no key.
4. Optional Featherless/AI-ML API model for polished synthesis. The default
   Featherless model is `deepseek-ai/DeepSeek-V3-0324`, matching the hackathon
   setup guide's sample call.

## Local test

From `backend/`:

```bash
python scripts/test_narrative_agent_local.py
```

Live ticker test:

```bash
python scripts/test_narrative_agent_local.py --live MSFT
```

If you do not have a real NewsAPI key, either delete `NEWS_API_KEY` from
`.env` or leave it as a placeholder. The agent will use free Yahoo/yfinance
fallbacks.

Run the Band-connected agent:

```bash
python -m agents.narrative_analyst.agent
```

If Band returns `401 Unauthorized`, fill in the real `narrative_analyst`
`agent_id` and `api_key` in `agent_config.yaml`. The example values are
placeholders and will not connect.
