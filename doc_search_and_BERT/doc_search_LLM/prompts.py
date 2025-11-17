"""
Prompt templates and utilities for rendering LLM prompts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from textwrap import dedent

from schemas import FewShotExample, QuestionSpec


@dataclass
class RenderedPrompt:
    """Container for a rendered prompt and bookkeeping metadata."""

    text: str
    truncated: bool
    truncated_chars: int
    prompt_template_id: str


FORMAT_INSTRUCTIONS = dedent(
    """\
    {
      "decision": "yes" | "no" | "unsure",
      "confidence": float between 0 and 1,
    #   "spans": [
    #     {
    #       "text": "Exact substring copied verbatim from the page content."
    #     }
    #   ],
    #   "rationale": "Short explanation under 80 words."
    }
    """
)


PROMPT_TEMPLATES: Dict[str, str] = {
    "baseline": dedent(
        """\
        You are an expert analyst reviewing annual report pages.
        Decide if the page answers the question using only the provided content.
        Be conservative: respond "no" when unsure.

        Question:
        {question}

        What counts as an answer:
        {include_section}

        What to ignore:
        {exclude_section}

        {few_shot_block}

        Page content (markdown):
        \"\"\"{page_content}\"\"\"

        Respond strictly in minified JSON matching the schema:
        {format_instructions}
        """
    ),
    "cot": dedent(
        """\
        You are verifying whether a report page answers a question.
        Think step by step before deciding.

        Question:
        {question}

        Inclusion criteria:
        {include_section}

        Exclusion criteria:
        {exclude_section}

        {few_shot_block}

        Page:
        \"\"\"{page_content}\"\"\"

        First briefly reason in JSON under the key "scratchpad" (max 60 words),
        then provide the final decision JSON under the key "answer".
        The "answer" value must match this schema:
        {format_instructions}
        """
    ),
    "extractive": dedent(
        """\
        You must perform extractive QA on the page.
        Copy supporting text exactly when the answer is present; otherwise state "no".

        Question:
        {question}

        Include:
        {include_section}

        Exclude:
        {exclude_section}

        {few_shot_block}

        Page markdown:
        \"\"\"{page_content}\"\"\"

        Output strict JSON following:
        {format_instructions}
        """
    ),
}


def format_bullets(items: Iterable[str]) -> str:
    """Return a human-readable bullet list, or 'None specified.'."""
    items = list(items)
    if not items:
        return "None specified."
    return "\n".join(f"- {item.strip()}" for item in items if item.strip())


def format_few_shot(examples: Iterable[FewShotExample]) -> str:
    """Render few-shot examples section."""
    rendered: List[str] = []
    for idx, ex in enumerate(examples, start=1):
        rendered.append(
            dedent(
                f"""\
                Example {idx} (label={ex.label.value}):
                Page excerpt:
                \"\"\"{ex.page_excerpt.strip()}\"\"\"
                Rationale: {ex.rationale.strip()}
                """
            ).strip()
        )
    if not rendered:
        return ""
    joined = "\n\n".join(rendered)
    return f"Few-shot guidance:\n{joined}\n\n"


def truncate_text(text: str, max_chars: Optional[int]) -> Tuple[str, bool, int]:
    """Truncate text to max_chars, attempting to keep headings."""
    if max_chars is None or len(text) <= max_chars:
        return text, False, 0
    truncated_text = text[:max_chars]
    return truncated_text, True, len(text) - len(truncated_text)


def render_prompt(
    question: QuestionSpec,
    page_content: str,
    *,
    prompt_template_id: str,
    few_shot: bool,
    truncate_at: Optional[int],
) -> RenderedPrompt:
    """Render the chosen prompt template for a question/page pair."""
    if prompt_template_id not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt template id: {prompt_template_id}")

    template = PROMPT_TEMPLATES[prompt_template_id]

    include_section = format_bullets(question.include)
    exclude_section = format_bullets(question.exclude)
    few_shot_examples = question.few_shot_examples if few_shot else []
    few_shot_block = format_few_shot(few_shot_examples)
    truncated_content, truncated, truncated_chars = truncate_text(page_content, truncate_at)
    prompt_text = template.format(
        question=question.question.strip(),
        include_section=include_section,
        exclude_section=exclude_section,
        few_shot_block=few_shot_block,
        page_content=truncated_content,
        format_instructions=FORMAT_INSTRUCTIONS,
    )
    return RenderedPrompt(
        text=prompt_text,
        truncated=truncated,
        truncated_chars=truncated_chars,
        prompt_template_id=prompt_template_id,
    )


def list_prompt_templates() -> List[str]:
    """Return sorted list of available template identifiers."""
    return sorted(PROMPT_TEMPLATES.keys())
