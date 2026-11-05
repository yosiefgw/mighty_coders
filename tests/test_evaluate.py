"""Tests for the pure helpers in python/evaluate.py and for eval/ground_truth.jsonl (no model needed)."""
import json
from pathlib import Path

import pytest

from evaluate import (DEFAULT_FILE, evaluate, first_hit_rank, overlap, percentile, read_questions, score,
                      usable_questions)


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


def test_question_kept_if_one_accepted_answer_exists():
    """Review PR #9: skip a question only when NONE of its answers is in the database."""
    questions = [
        {"id": 4, "question": "Olympus Mons?", "answers": ["Mars", "List of tallest mountains in the Solar System"]},
        {"id": 9, "question": "Heart-shaped region?", "answers": ["Pluto"]},
    ]
    kept, skipped, dropped = usable_questions(questions, titles={"Mars"})
    assert [q["id"] for q in kept] == [4]
    assert kept[0]["answers"] == ["Mars"]                       # only answers that exist are scored
    assert skipped == [9]
    assert dropped == [(4, "List of tallest mountains in the Solar System"), (9, "Pluto")]


def test_usable_questions_does_not_change_the_input():
    questions = [{"id": 1, "question": "Q?", "answers": ["A", "B"]}]
    usable_questions(questions, titles={"A"})
    assert questions[0]["answers"] == ["A", "B"]


def test_empty_database_skips_everything():
    kept, skipped, _ = usable_questions([{"id": 1, "question": "Q?", "answers": ["A"]}], titles=set())
    assert kept == [] and skipped == [1]


def test_evaluate_refuses_no_questions():
    """Review PR #9: an empty question list gave 'StatisticsError: no median for empty data'."""
    with pytest.raises(ValueError, match="no questions"):
        evaluate(conn=None, questions=[], vectors=[], k=5, ef_values=[20], repeat=1)
