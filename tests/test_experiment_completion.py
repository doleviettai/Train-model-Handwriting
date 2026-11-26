import json
import os
import tempfile
import unittest
from pathlib import Path

from src.experiments.notify_701515 import RUNS, completion_status, process_alive, read_json


class ExperimentCompletionTests(unittest.TestCase):
    def test_waiting_until_both_models_and_final_evaluation_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(any(row["ready"] for row in completion_status(root).values()))
            for kind, cfg in RUNS.items():
                folder = root / cfg["folder"]
                folder.mkdir(parents=True)
                (folder / cfg["model"]).write_bytes(b"example")
                (folder / cfg["state"]).write_text(json.dumps({"status": "running", "epoch": 4}))
                (folder / "pipeline_result.json").write_text(json.dumps({"status": "completed"}))
                # Stale 'completed' from a smoke run must not mark active training ready.
                self.assertFalse(completion_status(root)[kind]["ready"])
                (folder / cfg["state"]).write_text(json.dumps({"status": "early_stopped", "stage": "finished"}))
                if kind == "ghep":
                    self.assertFalse(completion_status(root)[kind]["ready"])
                    (folder / "shape_svm.joblib").write_bytes(b"example")
                    (folder / "hybrid_selection.json").write_text("{}")
                self.assertTrue(completion_status(root)[kind]["ready"])
            self.assertTrue(all(row["ready"] for row in completion_status(root).values()))

    def test_does_not_claim_ghep_ready_during_svm_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = RUNS["ghep"]
            folder = root / cfg["folder"]
            folder.mkdir(parents=True)
            for name in [cfg["model"], "shape_svm.joblib", "hybrid_selection.json"]:
                (folder / name).write_text("{}")
            (folder / "pipeline_result.json").write_text('{"status":"completed"}')
            (folder / cfg["state"]).write_text('{"status":"completed","stage":"fit_svm"}')
            self.assertFalse(completion_status(root)["ghep"]["ready"])

    def test_incomplete_status_is_waiting_not_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            self.assertEqual(read_json(path), {})
            path.write_text('{"status":')
            self.assertEqual(read_json(path), {})
        self.assertTrue(process_alive(os.getpid()))
        for pid in (None, 0, -1):
            self.assertFalse(process_alive(pid))


if __name__ == "__main__":
    unittest.main()
