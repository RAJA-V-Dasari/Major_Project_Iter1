"""
The stitch, end to end, on synthetic data: no student page is involved.

    python -m unittest discover -s tests -t .
    python pipeline.py check                    # runs these too

One booklet of three content pages is drawn with numpy - printed rules,
a margin, strokes standing in for handwriting - and given transcriptions
shaped like the reader's real output. Then each stage runs as a script,
in a temporary data root, exactly as `python pipeline.py run` runs it:

    build_booklet.py --all  ->  export.py  ->  load_handoff.py --report
                            ->  grade.py --all --no-semantic  ->  agreement.py

and the assertions are the properties that matter downstream:

  * a page carrying two questions is split at the heading, not filed
    whole under the last one (the bug this layout fixed);
  * a page continuing an answer is filed under the question it continues;
  * struck-out work reaches marking as `crossed`, excluded, and out of
    the prose that is scored;
  * a page that was never read reaches marking as a gap in the question
    that was open, so that question cannot be zeroed for it;
  * marking reads the handoff's totals back exactly as they were written.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

try:
    import cv2
    import numpy as np
except ImportError:                                  # pragma: no cover
    cv2 = None

ENGINE = "synthetic"

PAGE_2 = """### 1)
A is the client because it performs the active open.
B is the server.

### 2a)
The RTO is the retransmission timeout ~~and it never changes~~ which adapts.
"""

PAGE_3 = """which is why a lost acknowledgment causes a resend.

![diagram: three way handshake]

### 2b)
The routing table holds the destination and the next hop.
"""


def draw_page(path, rows, box=None):
    """A ruled answer sheet with stroke rows, and optionally a drawn box."""

    image = np.full((2177, 1598), 255, np.uint8)
    for y in range(240, 2120, 58):
        cv2.line(image, (60, y), (1540, y), 160, 1)          # printed rules
    cv2.line(image, (210, 120), (210, 2140), 160, 1)         # margin rule
    for row in rows:
        y = 240 + 58 * row
        for x in range(260, 1320, 64):
            cv2.line(image, (x, y - 34), (x + 36, y - 6), 20, 3)
    if box:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), 20, 3)
        cv2.line(image, (x1, (y1 + y2) // 2), (x2, (y1 + y2) // 2), 20, 3)
        cv2.line(image, ((x1 + x2) // 2, y1), ((x1 + x2) // 2, y2), 20, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


@unittest.skipIf(cv2 is None, "needs numpy and opencv-python-headless")
class Stitch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.data = Path(cls.tmp.name) / "data"

        pages = cls.data / "pages" / "student_01" / "cie_2"
        draw_page(pages / "page_02.png", rows=(1, 2, 3, 6, 7))
        draw_page(pages / "page_03.png", rows=(1, 2, 11),
                  box=(400, 420, 1200, 900))
        draw_page(pages / "page_04.png", rows=(1, 2))    # never read

        read = cls.data / "read" / ENGINE
        read.mkdir(parents=True)
        (read / "s01_c2_p02.md").write_text(PAGE_2, encoding="utf-8")
        (read / "s01_c2_p03.md").write_text(PAGE_3, encoding="utf-8")

        cls.env = dict(os.environ, MP_DATA=str(cls.data), PYTHONUTF8="1")
        for variable in ("MPE_HANDOFF", "MPE_SCHEMES", "MP_ENGINE"):
            cls.env.pop(variable, None)

        cls.run_script("reading/assemble/build_booklet.py",
                       "--all", "--engine", ENGINE)
        cls.run_script("reading/handoff/export.py")

        cls.booklet_dir = cls.data / "booklets" / "student_01_cie_2"
        cls.markdown = (cls.booklet_dir / "booklet.md").read_text(
            encoding="utf-8")
        cls.structure = json.loads(
            (cls.booklet_dir / "structure.json").read_text(encoding="utf-8"))
        cls.handoff = json.loads(
            (cls.data / "handoff" / "student_01_cie_2" / "booklet.json")
            .read_text(encoding="utf-8"))
        cls.parts = {p["label"]: p for q in cls.handoff["questions"]
                     for p in q["parts"]}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def run_script(cls, path, *args):
        result = subprocess.run([sys.executable, str(REPO / path), *args],
                                cwd=REPO, env=cls.env, capture_output=True,
                                text=True, encoding="utf-8")
        if result.returncode:
            raise AssertionError(f"{path} {' '.join(args)} failed:\n"
                                 f"{result.stdout}\n{result.stderr}")
        return result.stdout

    def section(self, label):
        """One `## <label>)` section of booklet.md."""

        text = self.markdown.split(f"## {label})", 1)[1]
        return text.split("\n## ", 1)[0]

    # --- assemble --------------------------------------------------------

    def test_a_page_with_two_questions_is_split_at_the_heading(self):
        self.assertIn("active open", self.section("1"))
        self.assertNotIn("active open", self.section("2a"))
        self.assertIn("retransmission timeout", self.section("2a"))

    def test_a_continuation_is_filed_under_the_question_it_continues(self):
        self.assertIn("lost acknowledgment", self.section("2a"))
        self.assertNotIn("lost acknowledgment", self.section("2b"))
        self.assertIn("routing table", self.section("2b"))

    def test_an_unread_page_is_a_gap_under_the_open_question(self):
        gaps = [e for e in self.structure["stream"] if "gap" in e]
        self.assertEqual([(g["page_id"], g["label"]) for g in gaps],
                         [("s01_c2_p04", "2b")])

    # --- handoff ---------------------------------------------------------

    def test_parts_keep_the_order_they_were_answered_in(self):
        self.assertEqual(list(self.parts), ["1", "2a", "2b"])
        self.assertEqual((self.parts["2a"]["number"],
                          self.parts["2a"]["part"]), ("2", "a"))

    def test_struck_out_work_is_carried_but_never_scored(self):
        answer = self.parts["2a"]["answer"]
        crossed = [i for i in answer if i["type"] == "crossed"]
        self.assertEqual([c["text"] for c in crossed],
                         ["and it never changes"])
        self.assertTrue(all(c["excluded"] for c in crossed))
        self.assertNotIn("never changes", self.parts["2a"]["text"])

    def test_the_gap_reaches_marking(self):
        gaps = [i for i in self.parts["2b"]["answer"] if i["type"] == "gap"]
        self.assertEqual([g["page_id"] for g in gaps], ["s01_c2_p04"])

    def test_a_continued_answer_records_both_pages(self):
        self.assertEqual(self.parts["2a"]["pages"],
                         ["s01_c2_p02", "s01_c2_p03"])
        self.assertTrue(self.parts["2a"]["crosses_page_break"])

    def test_every_diagram_points_at_a_file_that_exists(self):
        folder = self.data / "handoff" / "student_01_cie_2"
        for part in self.parts.values():
            for item in part["answer"]:
                if item["type"] == "diagram":
                    self.assertTrue((folder / item["image"]).exists(),
                                    item["image"])

    # --- marking ---------------------------------------------------------

    def test_marking_reads_the_handoff_totals_back_exactly(self):
        out = self.run_script("marking/src/load_handoff.py", "--report")
        self.assertIn("OK - we read the contract the same way", out)

    def test_the_ladder_and_the_report_run_on_it(self):
        self.run_script("marking/src/grade.py", "--all", "--no-semantic")
        marks = json.loads((self.data / "marking" / "marks" /
                            "student_01_cie_2.json").read_text(
                                encoding="utf-8"))
        by_id = {q["id"]: q for q in marks["questions"]}
        self.assertTrue(by_id["1"]["attempted"])
        self.assertIn("active open", by_id["1"]["answer"])
        self.assertEqual(by_id["2b"]["gaps"], 1)
        self.run_script("marking/src/agreement.py")
        self.assertTrue((self.data / "marking" / "agreement.md").exists())


class Units(unittest.TestCase):

    def test_an_unlabelled_answer_is_aligned_against_every_question(self):
        sys.path.insert(0, str(REPO / "marking" / "src"))
        import align
        valid = ["1", "2a", "2b"]
        self.assertEqual(align.candidates_from_label("", valid), valid)
        self.assertEqual(align.candidates_from_label(None or "", valid),
                         valid)
        self.assertEqual(align.candidates_from_label("2", valid),
                         ["2a", "2b"])

    def test_an_older_handoff_nested_under_data_is_found(self):
        from common import layout
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "handoff1" / "data"
            nested.mkdir(parents=True)
            (nested / "index.json").write_text("{}", encoding="utf-8")
            previous = os.environ.get("MPE_HANDOFF")
            os.environ["MPE_HANDOFF"] = str(Path(tmp) / "handoff1")
            try:
                self.assertEqual(layout.handoff_data(), nested.resolve())
            finally:
                if previous is None:
                    os.environ.pop("MPE_HANDOFF")
                else:
                    os.environ["MPE_HANDOFF"] = previous


if __name__ == "__main__":
    unittest.main()
