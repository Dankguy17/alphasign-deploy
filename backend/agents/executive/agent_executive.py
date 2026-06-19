"""
backend/executive.py

Executive LLM — reads the full conversation log, synthesises a structured
financial intelligence report, and renders it as a PDF using reportlab.

Called by main.py's SessionState._generate_report() once the turn limit is
reached.  Can also be invoked directly for testing:

    python executive.py alphasign_conversation.txt alphasign_report.pdf

Environment variables:
    GROQ_API_KEY                Required.
    EXECUTIVE_MODEL             Default: llama-3.3-70b-versatile
    GROQ_BASE_URL               Default: https://api.groq.com/openai/v1
    EXECUTIVE_MAX_TOKENS        Default: 4096
"""

from __future__ import annotations

import json
import logging
import os
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("alphasign.executive")

DEFAULT_GROQ_BASE_URL  = "https://api.groq.com/openai/v1"
DEFAULT_EXECUTIVE_MODEL = "llama-3.3-70b-versatile"
EXECUTIVE_MAX_TOKENS    = int(os.getenv("EXECUTIVE_MAX_TOKENS", "4096"))


# ── Groq synthesis ────────────────────────────────────────────────────────────

def _call_groq(conversation_text: str) -> str:
    """
    Send the full conversation log to Groq and return the Executive summary.
    Uses the openai-compatible SDK (langchain_openai / openai) that's already
    in the project's requirements.
    """
    from openai import OpenAI

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        raise RuntimeError("GROQ_API_KEY is required for Executive report generation.")

    model = (
        os.getenv("EXECUTIVE_MODEL")
        or os.getenv("GROQ_MODEL")
        or DEFAULT_EXECUTIVE_MODEL
    )
    base_url = os.getenv("GROQ_BASE_URL", DEFAULT_GROQ_BASE_URL)

    client = OpenAI(api_key=api_key, base_url=base_url)

    system_prompt = textwrap.dedent("""
        You are the Executive agent in AlphaSign, a multi-agent financial risk
        intelligence system. You have been given the complete conversation log
        produced by three specialist agents:

          • Narrative Analyst  — news research, narrative radar, source reliability
          • Signal Processing  — quantitative metrics (log returns, volatility, beta,
                                 idiosyncratic vol, market-adjusted return)
          • Latent State       — Kalman filter regime detection and trend prediction

        Your task is to synthesise their findings into a concise, professional
        Executive Intelligence Report. Structure your output EXACTLY as follows,
        using these section headers verbatim (they are used for PDF rendering):

        ## Executive Summary
        (2-3 paragraphs: what was investigated, the core finding, and the key risk)

        ## Asset Overview
        (Bullet list: ticker(s), date range covered, sentiment label, source reliability)

        ## Key Quantitative Findings
        (Numbered list of the most important metrics and what they imply)

        ## Narrative Signals
        (Bullet list: top themes, catalysts, risk flags from the Narrative Analyst)

        ## Regime & Trend Assessment
        (Paragraph: Kalman filter output — filtered level, trend slope, regime shift flag,
        next-value prediction, z-score, and what this implies for the current thesis)

        ## Hypothesis Verdict
        (Was the original narrative hypothesis supported by the quant and latent state
        evidence? Give a direct verdict: Supported / Weakly Supported / Not Supported,
        with one paragraph of reasoning citing actual numbers.)

        ## Risk Factors
        (Numbered list: top 3-5 risks identified across all three agents)

        ## Recommended Next Steps
        (Bulleted list of 3-5 concrete actions or further research questions)

        ---
        IMPORTANT RULES:
        - Do not invent numbers. Only cite figures that appear in the conversation.
        - Use precise financial language. Avoid vague phrases like "may" or "could potentially".
        - Be direct. Each section should be dense with evidence, not filler prose.
        - If a section cannot be completed because the agents did not produce the
          relevant output, say so explicitly rather than inventing content.
    """).strip()

    logger.info("Calling Groq Executive model: %s (%d tokens max)", model, EXECUTIVE_MAX_TOKENS)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"CONVERSATION LOG:\n\n{conversation_text}"},
        ],
        temperature=0.15,
        max_tokens=EXECUTIVE_MAX_TOKENS,
    )
    return response.choices[0].message.content or ""


# ── PDF rendering ─────────────────────────────────────────────────────────────

def _render_pdf(report_text: str, output_path: str) -> bytes:
    """
    Convert the Executive LLM output to a styled PDF using reportlab.
    Returns the raw PDF bytes AND writes to output_path.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, HRFlowable, PageBreak,
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        topMargin=1.0 * inch,
        bottomMargin=1.0 * inch,
    )

    base_styles = getSampleStyleSheet()

    # Custom styles
    styles = {
        "cover_title": ParagraphStyle(
            "cover_title",
            parent=base_styles["Title"],
            fontSize=26,
            leading=32,
            textColor=colors.HexColor("#0d1b2a"),
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub",
            parent=base_styles["Normal"],
            fontSize=11,
            textColor=colors.HexColor("#4a6fa5"),
            alignment=TA_CENTER,
            spaceAfter=24,
        ),
        "section_heading": ParagraphStyle(
            "section_heading",
            parent=base_styles["Heading1"],
            fontSize=13,
            leading=18,
            textColor=colors.HexColor("#0d1b2a"),
            spaceBefore=18,
            spaceAfter=6,
            borderPad=0,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base_styles["Normal"],
            fontSize=10,
            leading=15,
            textColor=colors.HexColor("#222222"),
            spaceAfter=8,
            alignment=TA_LEFT,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base_styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#222222"),
            leftIndent=18,
            bulletIndent=6,
            spaceAfter=4,
        ),
        "number": ParagraphStyle(
            "number",
            parent=base_styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#222222"),
            leftIndent=18,
            bulletIndent=6,
            spaceAfter=4,
        ),
        "footer": ParagraphStyle(
            "footer",
            parent=base_styles["Normal"],
            fontSize=8,
            textColor=colors.HexColor("#888888"),
            alignment=TA_CENTER,
        ),
    }

    story = []
    ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── Cover ──────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("AlphaSign", styles["cover_title"]))
    story.append(Paragraph("Executive Intelligence Report", styles["cover_sub"]))
    story.append(Paragraph(f"Generated: {ts_str}", styles["cover_sub"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#4a6fa5")))
    story.append(Spacer(1, 0.3 * inch))

    # ── Body: parse section headers and body text ─────────────────────────
    # Sections start with "## <Title>"; everything else is body text.
    # Lines starting with "- " or "* " become bullets.
    # Lines starting with a digit and ". " become numbered items.

    def _para(
        text: str,
        style_key: str,
        *,
        bullet_text: str | None = None,
    ) -> Paragraph:
        # Escape ampersands and angle brackets for ReportLab XML
        safe = (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
        )
        return Paragraph(safe, styles[style_key], bulletText=bullet_text)

    for raw_line in report_text.splitlines():
        line = raw_line.strip()
        if not line:
            story.append(Spacer(1, 4))
            continue

        if line.startswith("## "):
            heading = line[3:].strip()
            story.append(HRFlowable(
                width="100%", thickness=0.5,
                color=colors.HexColor("#cccccc"), spaceAfter=2,
            ))
            story.append(_para(heading, "section_heading"))

        elif line.startswith(("- ", "* ", "• ")):
            content = line[2:].strip()
            story.append(_para(content, "bullet", bullet_text="•"))

        elif len(line) > 2 and line[0].isdigit() and line[1] in ".)" and line[2] == " ":
            story.append(_para(line, "number"))

        elif line.startswith("---"):
            story.append(HRFlowable(
                width="100%", thickness=0.5,
                color=colors.HexColor("#cccccc"), spaceAfter=6,
            ))

        else:
            story.append(_para(line, "body"))

    # ── Footer note ───────────────────────────────────────────────────────
    story.append(Spacer(1, 0.4 * inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 6))
    story.append(_para(
        "This report was generated automatically by AlphaSign. "
        "It is for informational purposes only and does not constitute financial advice.",
        "footer",
    ))

    doc.build(story)

    return Path(output_path).read_bytes()


# ── Public entry point ────────────────────────────────────────────────────────

def generate_executive_report(conversation_text: str, output_path: str) -> bytes:
    """
    End-to-end: call Groq to synthesise the report, render to PDF, return bytes.

    This is a synchronous function; main.py calls it via asyncio.to_thread().
    """
    if not conversation_text.strip():
        raise ValueError("Conversation log is empty — nothing to report.")

    logger.info("Executive: synthesising report from %d chars of conversation…",
                len(conversation_text))
    report_text = _call_groq(conversation_text)

    if not report_text.strip():
        raise RuntimeError("Groq returned an empty Executive report.")

    logger.info("Executive: rendering PDF to %s…", output_path)
    pdf_bytes = _render_pdf(report_text, output_path)
    logger.info("Executive: PDF complete (%d bytes).", len(pdf_bytes))
    return pdf_bytes


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
    logging.basicConfig(level=logging.INFO)

    conv_path = sys.argv[1] if len(sys.argv) > 1 else "alphasign_conversation.txt"
    pdf_path  = sys.argv[2] if len(sys.argv) > 2 else "alphasign_report.pdf"

    conv_text = Path(conv_path).read_text(encoding="utf-8")
    pdf_bytes = generate_executive_report(conv_text, pdf_path)
    print(f"PDF written to {pdf_path} ({len(pdf_bytes):,} bytes)")
