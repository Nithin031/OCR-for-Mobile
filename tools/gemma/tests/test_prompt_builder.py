"""Unit tests for prompt_builder (stdlib only; token counts come from a fake counter)."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from prompt_builder import (  # noqa: E402
    OcrLine,
    PromptConfig,
    PromptError,
    build_prompt,
    fill,
    number_lines,
)

CONFIG = HERE.parents[1] / "reference" / "gemma" / "gemma_config.json"


def char_counter(messages):
    """Fake tokenizer: one token per 4 characters of all message contents, rounded up."""
    total = sum(len(m["content"]) for m in messages)
    return -(-total // 4)


class ConfigTest(unittest.TestCase):
    def test_loads_real_config(self):
        cfg = PromptConfig.from_config_file(CONFIG)
        self.assertEqual(cfg.low_confidence_threshold, 0.8)
        self.assertEqual(cfg.n_ctx, 4096)
        self.assertEqual(cfg.prompt_token_budget, cfg.n_ctx - cfg.max_new_tokens)
        self.assertIn("{lines}", cfg.user_template)
        self.assertIn("{question}", cfg.user_template)
        self.assertIn("{truncation_note}", cfg.user_template)
        self.assertIn("{omitted}", cfg.truncation_note_template)


class NumberingTest(unittest.TestCase):
    def setUp(self):
        self.cfg = PromptConfig.from_config_file(CONFIG)

    def test_numbering_in_given_order(self):
        numbered, line_map = number_lines(
            [OcrLine("Form 1", 0.99), OcrLine("Name", 0.95), OcrLine("Date of birth", 0.97)], self.cfg)
        self.assertEqual(numbered, ["[L1] Form 1", "[L2] Name", "[L3] Date of birth"])
        self.assertEqual(line_map, [0, 1, 2])

    def test_low_confidence_marking(self):
        numbered, _ = number_lines(
            [OcrLine("a", 0.79), OcrLine("b", 0.80), OcrLine("c", None), OcrLine("d", 0.0)], self.cfg)
        self.assertEqual(numbered, [
            "[L1] a (low confidence)",
            "[L2] b",
            "[L3] c",
            "[L4] d (low confidence)",
        ])

    def test_blank_lines_skipped_and_mapped(self):
        numbered, line_map = number_lines(
            [OcrLine("first", 0.9), OcrLine("", 0.0), OcrLine("   ", 0.5), OcrLine("fourth", 0.9)], self.cfg)
        self.assertEqual(numbered, ["[L1] first", "[L2] fourth"])
        self.assertEqual(line_map, [0, 3])

    def test_newlines_inside_a_line_become_spaces(self):
        numbered, _ = number_lines([OcrLine("  two\nparts\r\nhere ", 0.9)], self.cfg)
        self.assertEqual(numbered, ["[L1] two parts here"])

    def test_inserted_text_is_not_rescanned(self):
        self.assertEqual(fill("A {text} B", {"text": "{question}"}), "A {question} B")
        numbered, _ = number_lines([OcrLine("price {n} and {text}", 0.9)], self.cfg)
        self.assertEqual(numbered, ["[L1] price {n} and {text}"])

    def test_unknown_placeholder_fails(self):
        with self.assertRaises(PromptError):
            fill("{nope}", {})


class BuildPromptTest(unittest.TestCase):
    def setUp(self):
        self.cfg = PromptConfig.from_config_file(CONFIG)

    def test_exact_user_message(self):
        built = build_prompt([OcrLine("FORM 1", 0.93), OcrLine("Yes/No", 0.5)],
                             "What does this form ask for?", self.cfg, char_counter)
        self.assertEqual(built.messages[0], {"role": "system", "content": self.cfg.system_text})
        self.assertEqual(built.messages[1]["role"], "user")
        self.assertEqual(
            built.messages[1]["content"],
            "Document lines:\n[L1] FORM 1\n[L2] Yes/No (low confidence)\n\nQuestion: What does this form ask for?")
        self.assertEqual((built.n_lines_total, built.n_lines_kept, built.n_lines_omitted), (2, 2, 0))
        self.assertEqual(built.prompt_version, self.cfg.version)

    def test_empty_ocr_result(self):
        built = build_prompt([], "Explain this document.", self.cfg, char_counter)
        self.assertEqual(
            built.messages[1]["content"],
            "Document lines:\n(No text was recognized in this document.)\n\nQuestion: Explain this document.")
        self.assertEqual((built.n_lines_total, built.n_lines_kept, built.line_map), (0, 0, []))

    def test_only_blank_lines_counts_as_empty(self):
        built = build_prompt([OcrLine("", 0.0), OcrLine(" ", 0.1)], "Explain this document.", self.cfg, char_counter)
        self.assertIn("(No text was recognized in this document.)", built.messages[1]["content"])

    def test_truncation_keeps_reading_order_prefix_and_reports_omitted(self):
        lines = [OcrLine(f"line number {i:03d} " + "x" * 40, 0.99) for i in range(200)]
        base = char_counter(build_prompt(lines[:0], "Q?", self.cfg, char_counter).messages)
        cfg = replace(self.cfg, n_ctx=base + 600, max_new_tokens=100)   # budget = base + 500
        built = build_prompt(lines, "Q?", cfg, char_counter)
        k = built.n_lines_kept
        self.assertTrue(0 < k < 200)
        self.assertEqual(built.n_lines_omitted, 200 - k)
        self.assertLessEqual(built.prompt_tokens, cfg.prompt_token_budget)
        self.assertEqual(built.prompt_tokens, char_counter(built.messages))
        user = built.messages[1]["content"]
        self.assertIn(f"[L{k}] line number {k - 1:03d}", user)
        self.assertNotIn(f"[L{k + 1}]", user)
        self.assertIn(f"(Note: the last {200 - k} lines of the document were left out", user)
        self.assertEqual(built.line_map, list(range(k)))
        # one more line must not fit (k is the largest prefix that fits)
        numbered_plus = [f"[L{i + 1}] line number {i:03d} " + "x" * 40 for i in range(k + 1)]
        user_plus = ("Document lines:\n" + "\n".join(numbered_plus)
                     + f"\n(Note: the last {200 - k - 1} lines of the document were left out because the document is too long.)"
                     + "\n\nQuestion: Q?")
        over = char_counter([{"role": "system", "content": cfg.system_text}, {"role": "user", "content": user_plus}])
        self.assertGreater(over, cfg.prompt_token_budget)

    def test_no_note_when_everything_fits(self):
        built = build_prompt([OcrLine("a", 0.9)], "Q?", self.cfg, char_counter)
        self.assertNotIn("(Note:", built.messages[1]["content"])

    def test_question_alone_too_long_fails(self):
        cfg = replace(self.cfg, n_ctx=200, max_new_tokens=100)
        with self.assertRaises(PromptError):
            build_prompt([OcrLine("a", 0.9)], "Q" * 5000, cfg, char_counter)

    def test_budget_must_be_positive(self):
        cfg = replace(self.cfg, n_ctx=100, max_new_tokens=100)
        with self.assertRaises(PromptError):
            build_prompt([OcrLine("a", 0.9)], "Q?", cfg, char_counter)


if __name__ == "__main__":
    unittest.main()
