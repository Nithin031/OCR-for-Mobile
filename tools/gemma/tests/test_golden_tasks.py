"""Checks the committed golden_llm/tasks.json against the golden OCR JSON (stdlib only)."""
import sys
import unittest
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from golden_tasks import quick_actions_from_config, resolve_tasks  # noqa: E402
from prompt_builder import PromptConfig  # noqa: E402

REPO = HERE.parents[2]
CONFIG = REPO / "tools" / "reference" / "gemma" / "gemma_config.json"
TASKS = REPO / "tools" / "reference" / "gemma" / "golden_llm" / "tasks.json"


class GoldenTasksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = PromptConfig.from_config_file(CONFIG)
        cls.tasks = resolve_tasks(TASKS, REPO, cls.cfg, quick_actions_from_config(CONFIG))

    def test_twenty_tasks_with_all_three_types(self):
        self.assertEqual(len(self.tasks), 20)
        counts = Counter(t.type for t in self.tasks)
        self.assertEqual(set(counts), {"summary", "field", "value"})
        for n in counts.values():
            self.assertGreaterEqual(n, 5)

    def test_line_placeholders_resolved(self):
        for t in self.tasks:
            self.assertNotIn("{line:", t.question, t.id)

    def test_known_resolutions(self):
        by_id = {t.id: t for t in self.tasks}
        # W-4 has one blank golden line (index 51), so golden index 101 is [L101]
        self.assertEqual(by_id["t14"].question, "What date appears in line [L101]?")
        self.assertEqual(by_id["t15"].question, "What phone number appears in line [L24]?")
        self.assertEqual(by_id["t01"].question, "Explain this document.")
        self.assertEqual(by_id["t02"].question, "What does this form ask for?")

    def test_value_tasks_point_at_lines(self):
        # resolve_tasks already fails if expected_text is not verbatim in the expected lines
        for t in self.tasks:
            if t.type == "value":
                self.assertTrue(t.expected_line_ids, t.id)
                self.assertTrue(t.expected_text or t.expect_not_in_document, t.id)


if __name__ == "__main__":
    unittest.main()
