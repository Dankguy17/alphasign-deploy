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
EXECUTIVE_TIMEOUT_SECONDS = float(os.getenv("EXECUTIVE_TIMEOUT_SECONDS", "90"))


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

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=EXECUTIVE_TIMEOUT_SECONDS,
        max_retries=1,
    )

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

        ## Strategy Recommendation
        (The first line MUST contain exactly one recommendation: BUY, SELL, or HOLD.
        On the next line, state the confidence as a whole-number percentage. Then give
        one concise paragraph explaining the strategy, entry/confirmation condition,
        invalidation condition, and time horizon using only available evidence.)

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
        - A BUY or SELL requires corroboration from both quantitative and latent-state
          evidence. Use HOLD when evidence is mixed or insufficient.
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
    import re
    from xml.sax.saxutils import escape
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether,
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER

    # Keep these values in sync with frontend/src/app/globals.css. ReportLab
    # renders in print/PDF color space, so translucent web colors are replaced
    # with visually equivalent solid colors.
    palette = {
        "canvas": colors.HexColor("#010102"),
        "surface_1": colors.HexColor("#0f1011"),
        "surface_2": colors.HexColor("#141516"),
        "hairline": colors.HexColor("#34343a"),
        "ink": colors.HexColor("#f7f8f8"),
        "muted": colors.HexColor("#d0d6e0"),
        "subtle": colors.HexColor("#8a8f98"),
        "tertiary": colors.HexColor("#62666d"),
        "primary": colors.HexColor("#5e6ad2"),
        "primary_hover": colors.HexColor("#828fff"),
        "positive": colors.HexColor("#2fbf71"),
        "negative": colors.HexColor("#eb5757"),
        "warning": colors.HexColor("#e2a336"),
    }

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=0.62 * inch,
        rightMargin=0.62 * inch,
        topMargin=0.68 * inch,
        bottomMargin=0.62 * inch,
    )

    base_styles = getSampleStyleSheet()

    # Custom styles
    styles = {
        "cover_title": ParagraphStyle(
            "cover_title",
            parent=base_styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=32,
            leading=36,
            textColor=palette["ink"],
            alignment=TA_LEFT,
            spaceAfter=8,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub",
            parent=base_styles["Normal"],
            fontSize=11,
            leading=16,
            textColor=palette["subtle"],
            alignment=TA_LEFT,
            spaceAfter=5,
        ),
        "section_heading": ParagraphStyle(
            "section_heading",
            parent=base_styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=palette["ink"],
            spaceAfter=0,
            borderPad=0,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base_styles["Normal"],
            fontSize=10,
            leading=15,
            textColor=palette["muted"],
            spaceAfter=7,
            alignment=TA_LEFT,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base_styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=palette["muted"],
            leftIndent=18,
            bulletIndent=6,
            spaceAfter=4,
        ),
        "number": ParagraphStyle(
            "number",
            parent=base_styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=palette["muted"],
            leftIndent=18,
            bulletIndent=6,
            spaceAfter=4,
        ),
        "footer": ParagraphStyle(
            "footer",
            parent=base_styles["Normal"],
            fontSize=7.5,
            leading=11,
            textColor=palette["tertiary"],
            alignment=TA_CENTER,
        ),
        "eyebrow": ParagraphStyle(
            "eyebrow", parent=base_styles["Normal"], fontName="Helvetica-Bold",
            fontSize=8, leading=10, textColor=palette["primary_hover"],
            spaceAfter=10,
        ),
        "page_number": ParagraphStyle(
            "page_number", parent=base_styles["Normal"], fontName="Helvetica",
            fontSize=7, textColor=palette["tertiary"], alignment=TA_CENTER,
        ),
        "strategy_label": ParagraphStyle(
            "strategy_label", parent=base_styles["Normal"], fontName="Helvetica-Bold",
            fontSize=8, leading=10, textColor=palette["subtle"], spaceAfter=4,
        ),
        "strategy_action": ParagraphStyle(
            "strategy_action", parent=base_styles["Normal"], fontName="Helvetica-Bold",
            fontSize=30, leading=34, textColor=palette["ink"], spaceAfter=4,
        ),
    }

    story = []
    ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── Cover ──────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.38 * inch))
    story.append(Paragraph("ALPHASIGN / EXECUTIVE INTELLIGENCE", styles["eyebrow"]))
    story.append(Paragraph("AlphaSign", styles["cover_title"]))
    story.append(Paragraph("Executive Intelligence Report", styles["cover_sub"]))
    story.append(Paragraph(f"Generated {ts_str}", styles["cover_sub"]))
    story.append(Spacer(1, 0.32 * inch))

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
        safe = escape(text)
        # Preserve the small amount of Markdown emphasis commonly returned by
        # the executive model without allowing arbitrary ReportLab markup.
        safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
        return Paragraph(safe, styles[style_key], bulletText=bullet_text)

    sections: list[tuple[str, list[tuple[str, str]]]] = []
    heading = "Executive Brief"
    content: list[tuple[str, str]] = []
    for raw_line in report_text.splitlines():
        line = raw_line.strip()
        if not line:
            if content and content[-1][0] != "space":
                content.append(("space", ""))
            continue
        if line.startswith("## "):
            if content:
                sections.append((heading, content))
            heading, content = line[3:].strip(), []
        elif line.startswith(("- ", "* ", "• ")):
            content.append(("bullet", line[2:].strip()))
        elif re.match(r"^\d+[.)]\s", line):
            content.append(("number", line))
        elif line.startswith("---"):
            continue
        else:
            content.append(("body", line))
    if content:
        sections.append((heading, content))

    # Keep the decision immediately below the title even when rendering reports
    # produced by an older prompt that placed this section later in the output.
    sections.sort(key=lambda section: section[0].lower() != "strategy recommendation")

    card_width = letter[0] - doc.leftMargin - doc.rightMargin
    for index, (section_title, items) in enumerate(sections):
        if section_title.lower() == "strategy recommendation":
            strategy_text = " ".join(value for kind, value in items if kind != "space")
            action_match = re.search(r"\b(BUY|SELL|HOLD)\b", strategy_text, re.IGNORECASE)
            action = action_match.group(1).upper() if action_match else "HOLD"
            action_color = {
                "BUY": palette["positive"],
                "SELL": palette["negative"],
                "HOLD": palette["warning"],
            }[action]
            details = []
            removed_action = False
            for kind, value in items:
                if not removed_action and re.fullmatch(
                    r"(?:\*\*)?\s*(BUY|SELL|HOLD)\s*(?:\*\*)?[.!]?",
                    value,
                    re.IGNORECASE,
                ):
                    removed_action = True
                    continue
                if kind == "space":
                    details.append(Spacer(1, 3))
                elif kind == "bullet":
                    details.append(_para(value, "bullet", bullet_text="•"))
                else:
                    details.append(_para(value, "body"))
            strategy_card = Table(
                [[
                    [
                        Paragraph("STRATEGY RECOMMENDATION", styles["strategy_label"]),
                        Paragraph(action, ParagraphStyle(
                            "strategy_action_colored",
                            parent=styles["strategy_action"],
                            textColor=action_color,
                        )),
                    ],
                    details,
                ]],
                colWidths=[1.72 * inch, card_width - 1.72 * inch],
                style=TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), palette["surface_1"]),
                    ("BOX", (0, 0), (-1, -1), 1.1, action_color),
                    ("LINEAFTER", (0, 0), (0, 0), 0.75, palette["hairline"]),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 16),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 16),
                    ("TOPPADDING", (0, 0), (-1, -1), 15),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ]),
            )
            story.append(KeepTogether([strategy_card, Spacer(1, 10)]))
            continue

        heading_row = Table(
            [["", _para(section_title, "section_heading")]],
            colWidths=[3, card_width - 3 - 26],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (0, 0), palette["primary"]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (0, 0), 0),
                ("RIGHTPADDING", (0, 0), (0, 0), 0),
                ("LEFTPADDING", (1, 0), (1, 0), 10),
                ("RIGHTPADDING", (1, 0), (1, 0), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]),
        )
        body_flowables = []
        for kind, value in items:
            if kind == "space":
                body_flowables.append(Spacer(1, 3))
            elif kind == "bullet":
                body_flowables.append(_para(value, "bullet", bullet_text="•"))
            else:
                body_flowables.append(_para(value, kind))
        card = Table(
            [[heading_row], [body_flowables]],
            colWidths=[card_width],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), palette["surface_1"]),
                ("BOX", (0, 0), (-1, -1), 0.75, palette["hairline"]),
                ("LEFTPADDING", (0, 0), (-1, -1), 16),
                ("RIGHTPADDING", (0, 0), (-1, -1), 16),
                ("TOPPADDING", (0, 0), (0, 0), 14),
                ("BOTTOMPADDING", (0, 0), (0, 0), 10),
                ("TOPPADDING", (0, 1), (0, 1), 3),
                ("BOTTOMPADDING", (0, 1), (0, 1), 10),
            ]),
            splitByRow=1,
            splitInRow=1,
        )
        story.append(KeepTogether([card, Spacer(1, 10)]) if index == 0 else card)
        story.append(Spacer(1, 10))

    # ── Footer note ───────────────────────────────────────────────────────
    story.append(Spacer(1, 8))
    story.append(_para(
        "This report was generated automatically by AlphaSign. "
        "It is for informational purposes only and does not constitute financial advice.",
        "footer",
    ))

    def _draw_page(canvas, built_doc):
        canvas.saveState()
        canvas.setFillColor(palette["canvas"])
        canvas.rect(0, 0, letter[0], letter[1], fill=1, stroke=0)
        # A thin lavender top rail carries the website's single accent color.
        canvas.setFillColor(palette["primary"])
        canvas.rect(0, letter[1] - 3, letter[0], 3, fill=1, stroke=0)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(palette["tertiary"])
        canvas.drawCentredString(letter[0] / 2, 18, f"ALPHASIGN  /  {built_doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_draw_page, onLaterPages=_draw_page)

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


def generate_fallback_executive_report(
    conversation_text: str, output_path: str, failure_reason: str
) -> bytes:
    """Render the collected evidence when LLM synthesis is unavailable."""
    excerpt = conversation_text[-12000:].strip()
    report_text = textwrap.dedent(f"""
        ## Strategy Recommendation
        HOLD
        Confidence unavailable. Automated Executive synthesis failed, so no
        evidence-backed trade recommendation can be issued.

        ## Executive Summary
        AlphaSign completed the specialist-agent analysis, but Executive report
        synthesis was unavailable. This fallback preserves the collected evidence
        for review instead of leaving the session without a report.

        ## Risk Factors
        - Executive synthesis failure: {failure_reason[:500]}
        - The transcript below is unsynthesised and requires human review.

        ## Agent Transcript
        {excerpt}
    """).strip()
    return _render_pdf(report_text, output_path)


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
