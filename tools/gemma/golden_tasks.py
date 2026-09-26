"""Loads golden_llm/tasks.json and the golden OCR lines they refer to."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from prompt_builder import OcrLine, PromptConfig, number_lines

_LINE_REF = re.compile(r"\{line:(\d+)\}")
TASK_TYPES = ("summary", "field", "value")


class TaskError(Exception):
    pass


@dataclass
class ResolvedTask:
    id: str
    type: str
    document: str
    question: str                 # placeholders resolved to [Ln]
    lines: list[OcrLine]          # golden OCR lines, reading order as stored
    expected_line_ids: list[int]  # [Ln] numbers
    expected_text: list[str]
    expect_not_in_document: bool
    checks: str


def load_golden_lines(golden_dir: Path, document: str) -> list[OcrLine]:
    path = golden_dir / f"{document}.json"
    if not path.is_file():
        raise TaskError(f"golden OCR file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return [OcrLine(text=l["text"], confidence=l.get("confidence")) for l in data["lines"]]


def resolve_tasks(tasks_file: Path, repo_root: Path, cfg: PromptConfig, quick_actions: dict[str, str]) -> list[ResolvedTask]:
    spec = json.loads(tasks_file.read_text(encoding="utf-8"))
    golden_dir = repo_root / spec["ocr_golden_dir"]
    out: list[ResolvedTask] = []
    seen: set[str] = set()
    for t in spec["tasks"]:
        tid = t["id"]
        if tid in seen:
            raise TaskError(f"duplicate task id {tid}")
        seen.add(tid)
        if t["type"] not in TASK_TYPES:
            raise TaskError(f"{tid}: unknown type {t['type']}")
        lines = load_golden_lines(golden_dir, t["document"])
        _, line_map = number_lines(lines, cfg)
        index_to_lid = {idx: i + 1 for i, idx in enumerate(line_map)}

        def lid(idx: int) -> int:
            if idx < 0 or idx >= len(lines):
                raise TaskError(f"{tid}: golden line {idx} does not exist in {t['document']} ({len(lines)} lines)")
            if idx not in index_to_lid:
                raise TaskError(f"{tid}: golden line {idx} of {t['document']} is blank, it has no [Ln] id")
            return index_to_lid[idx]

        if "quick_action" in t:
            if "question" in t:
                raise TaskError(f"{tid}: give either quick_action or question, not both")
            if t["quick_action"] not in quick_actions:
                raise TaskError(f"{tid}: quick_action {t['quick_action']} not in gemma_config.json")
            question = quick_actions[t["quick_action"]]
        else:
            question = _LINE_REF.sub(lambda m: f"[L{lid(int(m.group(1)))}]", t["question"])

        expected_idx = t.get("expected_lines", [])
        expected_ids = [lid(i) for i in expected_idx]
        expected_text = t.get("expected_text", [])
        for s in expected_text:
            if not any(s in lines[i].text for i in expected_idx):
                raise TaskError(f"{tid}: expected text {s!r} is not verbatim in the expected lines of {t['document']}")
        out.append(ResolvedTask(
            id=tid, type=t["type"], document=t["document"], question=question, lines=lines,
            expected_line_ids=expected_ids, expected_text=expected_text,
            expect_not_in_document=bool(t.get("expect_not_in_document", False)),
            checks=t.get("checks", ""),
        ))
    return out


def quick_actions_from_config(config_file: Path) -> dict[str, str]:
    cfg = json.loads(config_file.read_text(encoding="utf-8"))
    return dict(cfg["prompt"]["quick_actions"]["value"])
