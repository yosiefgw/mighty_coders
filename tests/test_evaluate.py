"""Tests for the pure helpers in python/evaluate.py and for eval/ground_truth.jsonl (no model needed)."""
import json
from pathlib import Path

import pytest

from evaluate import DEFAULT_FILE, first_hit_rank, overlap, percentile, read_questions, score


def test_first_hit_rank():
    assert first_hit_rank(["Mars", "Jupiter", "Jupiter"], ["Jupiter"]) == 2
    assert first_hit_rank(["Mars", "Venus"], ["Jupiter"]) is None
    assert first_hit_rank(["Saturn"], ["Saturn's hexagon", "Saturn"]) == 1   # any accepted title counts


def test_score():
    s = score([1, 2, None, 5], k=5)
    assert s["hit@1"] == 0.25
    assert s["hit@5"] == 0.75
    assert s["mrr"] == pytest.approx((1 + 0.5 + 0.2) / 4)


def test_score_rank_beyond_k_is_a_miss():
    assert score([6], k=5)["hit@5"] == 0.0


def test_overlap():
    assert overlap([1, 2, 3], [1, 2, 4]) == pytest.approx(2 / 3)
    assert overlap([], []) == 1.0


def test_percentile():
    values = list(range(1, 101))      # 1..100
    assert percentile(values, 95) == 95
    assert percentile([7.0], 95) == 7.0


def test_read_questions_rejects_missing_answers(tmp_path):
    p = tmp_path / "gt.jsonl"
    p.write_text(json.dumps({"id": 1, "question": "Q?", "answers": []}) + "\n")
    with pytest.raises(ValueError):
        read_questions(p)


def test_ground_truth_file_is_valid():
    questions = read_questions(DEFAULT_FILE)
    assert len(questions) >= 30
    assert len({q["id"] for q in questions}) == len(questions)        # unique ids
    assert len({q["question"] for q in questions}) == len(questions)  # no duplicate questions
    for q in questions:
        assert q["question"].endswith("?")
        assert all(isinstance(a, str) and a for a in q["answers"])
