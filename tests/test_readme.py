from pathlib import Path

SERIES = [
    "ai-lab-pyspark-sales-aggregation",
    "ai-lab-pyspark-price-regression",
    "ai-lab-pyspark-demand-forecast",
    "ai-lab-pyspark-churn-scoring",
    "ai-lab-pyspark-anomaly-detection",
]
ROOT = Path(__file__).resolve().parent.parent


def test_series_table_in_both_readmes():
    for name in ("README.md", "README.ko.md"):
        lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
        text = "\n".join(line for line in lines if line.startswith("|"))  # the series table only
        positions = [text.index(repo) for repo in SERIES]  # raises ValueError if one is missing
        assert positions == sorted(positions), f"{name}: series must be listed in order"
