import json
import random

import pytest

from churn_scoring.threshold import best_threshold, load_threshold, profit, save_threshold

PAIRS = [(0.9, 1.0), (0.8, 1.0), (0.7, 0.0), (0.4, 1.0), (0.3, 0.0), (0.2, 0.0), (0.1, 0.0), (0.05, 0.0)]


def test_profit_hand_computed():
    # 60 true churners reached x 50 value x 0.3 success = 900; 160 contacts x 5 = 800
    assert profit(60, 100, contact_cost=5, save_value=50, success_rate=0.3) == pytest.approx(100.0)
    assert profit(0, 0, 5, 50, 0.3) == 0.0
    assert profit(0, 10, 5, 50, 0.3) == -50.0


def test_best_threshold_by_hand():
    # value 100 x rate 0.5 = 50 per hit, contact cost 10
    # t=0.8: 2 contacts, 2 hits -> 100 - 20 = 80 ; t=0.4: 4 contacts, 3 hits -> 150 - 40 = 110
    best = best_threshold(PAIRS, contact_cost=10, save_value=100, success_rate=0.5, grid=[0.8, 0.4, 0.05])
    assert best["threshold"] == 0.4 and best["profit"] == pytest.approx(110.0)


def test_threshold_moves_with_contact_cost():
    rng = random.Random(3)
    pairs = []
    for _ in range(600):
        label = 1.0 if rng.random() < 0.15 else 0.0
        pairs.append((min(0.99, max(0.01, rng.gauss(0.6 if label else 0.25, 0.2))), label))
    last = 0.0
    for cost in (0, 1, 5, 10, 20, 40):
        chosen = best_threshold(pairs, cost, 50, 0.3)["threshold"]
        assert chosen >= last
        last = chosen
    assert best_threshold(pairs, 40, 50, 0.3)["threshold"] > best_threshold(pairs, 0, 50, 0.3)["threshold"]


def test_save_and_load_threshold(tmp_path):
    path = tmp_path / "models" / "threshold.json"
    assert load_threshold(path) == 0.5  # default when nothing is saved
    save_threshold(path, {"threshold": 0.66, "profit": 1.0}, {"contact_cost": 5})
    assert load_threshold(path) == 0.66
    assert json.loads(path.read_text())["contact_cost"] == 5
