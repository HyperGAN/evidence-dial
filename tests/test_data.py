import random
from evidence_dial.data import family_rows, verify_source


def test_all_declared_families_have_four_verified_cases():
    rng = random.Random(20261110)
    for i in range(112):
        rows = family_rows(f"P{i:03d}", "dev", rng)
        assert {r["case_type"] for r in rows} == {"missing", "conflict", "answerable", "altered"}
        for row in rows:
            verify_source(row)
