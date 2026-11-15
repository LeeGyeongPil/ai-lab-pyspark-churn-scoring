"""One place that builds the SparkSession (temp folders, quiet logs, fixed partitions)."""

from __future__ import annotations

import os
from pathlib import Path

from .javacheck import ensure_java


def tmp_dir_from_env() -> Path:
    return Path(os.environ.get("CHURN_TMP_DIR") or ".spark-tmp").resolve()


def get_spark(threads: str = "*", shuffle_partitions: int = 8, tmp_dir: Path | None = None):
    """Start (or reuse) a local SparkSession.

    Temp folders go inside the project because some sandboxes forbid the system
    temp folder, which makes the JVM die with JAVA_GATEWAY_EXITED.
    """
    ensure_java()
    tmp = (tmp_dir or tmp_dir_from_env()).resolve()
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["SPARK_LOCAL_DIRS"] = str(tmp)
    flag = f"-Djava.io.tmpdir={tmp}"
    existing = os.environ.get("JAVA_TOOL_OPTIONS", "")
    if "-Djava.io.tmpdir=" not in existing:
        os.environ["JAVA_TOOL_OPTIONS"] = f"{existing} {flag}".strip()

    from pyspark.sql import SparkSession  # imported late: Java must be ready first

    spark = (
        SparkSession.builder.master(f"local[{threads}]")
        .appName("churn-scoring")
        .config("spark.local.dir", str(tmp))
        .config("spark.sql.warehouse.dir", str(tmp / "warehouse"))
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
