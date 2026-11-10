from evidence_dial.evaluate import evaluate_rows
from evidence_dial.io import read_jsonl
from evidence_dial.protocol import digest


class FakeGenerator:
    last = {"seconds": 0.01, "output_tokens": 1}
    calls = 0
    capture_path = None

    def __call__(self, texts):
        self.calls += 1
        return ["UNKNOWN" for _ in texts]


def test_auditkit_callable_scores_and_resume_skips_committed(tmp_path):
    rows = [{"id": "test-missing", "family": "test", "dataset": "policy", "case_type": "missing",
             "context": "No refund policy is supplied.", "question": "Refund period?", "gold": "UNKNOWN", "aliases": ["UNKNOWN"]}]
    gen = FakeGenerator()
    summary = evaluate_rows(rows, tmp_path, gen)
    assert summary["success_rate"] == 1 and summary["complete"]
    calls = gen.calls
    evaluate_rows(rows, tmp_path, gen)
    assert gen.calls == calls
    assert len(read_jsonl(tmp_path / "predictions.jsonl")) == 1
