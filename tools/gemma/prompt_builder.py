"""Builds assistant prompts from OCR lines, exactly as Phase 7a will on the phone.

Every text and threshold comes from gemma_config.json. The Kotlin port in
llm-core must produce byte-identical messages for the same OCR lines; the
golden_llm fixtures written by g0_pipeline.py are the reference for that.

The chat template itself is not applied here. Token counting is delegated to
the caller (`count_tokens(messages)`), which in G0 is the C++ runner doing
llama_chat_apply_template + llama_tokenize with the pinned llama.cpp.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")


class PromptError(Exception):
    pass


@dataclass(frozen=True)
class OcrLine:
    text: str
    confidence: float | None


@dataclass(frozen=True)
class PromptConfig:
    version: str
    system_text: str
    line_template: str
    low_confidence_threshold: float
    low_confidence_suffix: str
    skip_blank_lines: bool
    user_template: str
    line_separator: str
    truncation_note_template: str
    empty_document_text: str
    n_ctx: int
    max_new_tokens: int

    @property
    def prompt_token_budget(self) -> int:
        return self.n_ctx - self.max_new_tokens

    @staticmethod
    def from_config_file(path: Path) -> "PromptConfig":
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
        p = cfg["prompt"]

        def v(section: dict, key: str):
            if key not in section or "value" not in section[key]:
                raise PromptError(f"gemma_config.json is missing {key}.value")
            return section[key]["value"]

        return PromptConfig(
            version=str(v(p, "version")),
            system_text=v(p, "system_text"),
            line_template=v(p, "line_template"),
            low_confidence_threshold=float(v(p, "low_confidence_threshold")),
            low_confidence_suffix=v(p, "low_confidence_suffix"),
            skip_blank_lines=bool(v(p, "skip_blank_lines")),
            user_template=v(p, "user_template"),
            line_separator=v(p, "line_separator"),
            truncation_note_template=v(p, "truncation_note_template"),
            empty_document_text=v(p, "empty_document_text"),
            n_ctx=int(v(cfg["runtime"], "n_ctx")),
            max_new_tokens=int(v(cfg["generation"], "max_new_tokens")),
        )


@dataclass
class BuiltPrompt:
    messages: list[dict[str, str]]
    prompt_version: str
    n_lines_total: int          # non-blank lines available
    n_lines_kept: int
    n_lines_omitted: int
    line_map: list[int]         # line_map[i] = OCR line index of [L{i+1}]
    prompt_tokens: int
    token_budget: int
    numbered_lines: list[str] = field(repr=False, default_factory=list)


def fill(template: str, values: dict[str, str]) -> str:
    """Replace {name} in the template only; inserted values are never re-scanned."""
    def repl(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in values:
            raise PromptError(f"template placeholder {{{name}}} has no value")
        return values[name]
    return _PLACEHOLDER.sub(repl, template)


def normalize_line_text(text: str) -> str:
    return text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").strip()


def number_lines(lines: Sequence[OcrLine], cfg: PromptConfig) -> tuple[list[str], list[int]]:
    numbered: list[str] = []
    line_map: list[int] = []
    for idx, line in enumerate(lines):
        text = normalize_line_text(line.text)
        if cfg.skip_blank_lines and text == "":
            continue
        n = len(numbered) + 1
        rendered = fill(cfg.line_template, {"n": str(n), "text": text})
        if line.confidence is not None and line.confidence < cfg.low_confidence_threshold:
            rendered += cfg.low_confidence_suffix
        numbered.append(rendered)
        line_map.append(idx)
    return numbered, line_map


def _messages(cfg: PromptConfig, numbered: list[str], k: int, question: str) -> list[dict[str, str]]:
    n = len(numbered)
    if n == 0:
        lines_block = cfg.empty_document_text
        note = ""
    else:
        lines_block = cfg.line_separator.join(numbered[:k])
        note = fill(cfg.truncation_note_template, {"omitted": str(n - k)}) if k < n else ""
    user = fill(cfg.user_template, {"lines": lines_block, "truncation_note": note, "question": question})
    return [{"role": "system", "content": cfg.system_text}, {"role": "user", "content": user}]


def build_prompt(
    lines: Sequence[OcrLine],
    question: str,
    cfg: PromptConfig,
    count_tokens: Callable[[list[dict[str, str]]], int],
) -> BuiltPrompt:
    numbered, line_map = number_lines(lines, cfg)
    n = len(numbered)
    budget = cfg.prompt_token_budget
    if budget <= 0:
        raise PromptError(f"n_ctx ({cfg.n_ctx}) must be larger than max_new_tokens ({cfg.max_new_tokens})")

    def fits(k: int) -> tuple[bool, int]:
        t = count_tokens(_messages(cfg, numbered, k, question))
        return t <= budget, t

    ok, tokens = fits(n)
    k = n
    if not ok:
        if n == 0:
            raise PromptError(f"prompt needs {tokens} tokens, budget is {budget}")
        lo, hi = 0, n - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if fits(mid)[0]:
                lo = mid
            else:
                hi = mid - 1
        ok, tokens = fits(lo)
        if not ok:
            raise PromptError(f"even with no document lines the prompt needs {tokens} tokens, budget is {budget}")
        k = lo

    return BuiltPrompt(
        messages=_messages(cfg, numbered, k, question),
        prompt_version=cfg.version,
        n_lines_total=n,
        n_lines_kept=k,
        n_lines_omitted=n - k,
        line_map=line_map[:k],
        prompt_tokens=tokens,
        token_budget=budget,
        numbered_lines=numbered[:k],
    )
