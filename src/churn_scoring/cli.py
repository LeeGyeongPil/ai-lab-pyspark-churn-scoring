"""Command line: `churn generate|features|train|evaluate|threshold|score|explain|all`."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

from . import __version__
from .i18n import LANGS, get_lang, resolve_lang, set_lang, t
from .layout import DEFAULT_CUTOFF, LABEL_DAYS, Layout

ACTIVE_DAYS = 30
COMMANDS = ("generate", "features", "train", "evaluate", "threshold", "score", "explain", "all")


# ---------------------------------------------------------------- argument parsing

def _preparse_lang(argv: list[str]) -> str:
    """Find --lang before the real parser exists, so --help can already be translated."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--lang", choices=LANGS)
    known, _ = pre.parse_known_args(argv)
    return resolve_lang(known.lang)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="churn", description=t("help.main"))
    parser.add_argument("--version", action="version", version=f"churn {__version__}")
    parser.add_argument("--lang", choices=LANGS, help=t("help.lang"))

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--lang", choices=LANGS, default=argparse.SUPPRESS, help=t("help.lang"))
    common.add_argument("--data-dir", type=Path, default=Path("data"), help=t("help.data_dir"))
    common.add_argument("--model-dir", type=Path, default=Path("models"), help=t("help.model_dir"))
    common.add_argument("--out-dir", type=Path, default=Path("out"), help=t("help.out_dir"))

    sub = parser.add_subparsers(dest="command", required=True, metavar="<step>")

    def add(name: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, parents=[common], help=t(f"help.{name}"), description=t(f"help.{name}"))

    def add_generate_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--customers", type=_positive_int, default=3000, help=t("help.customers"))
        p.add_argument("--seed", type=int, default=42, help=t("help.seed"))

    def add_cutoff(p: argparse.ArgumentParser) -> None:
        p.add_argument("--cutoff", type=_iso_date, default=None, help=t("help.cutoff"))

    def add_money(p: argparse.ArgumentParser) -> None:
        p.add_argument("--contact-cost", type=_non_negative, default=5.0, help=t("help.contact_cost"))
        p.add_argument("--save-value", type=_non_negative, default=200.0, help=t("help.save_value"))
        p.add_argument("--success-rate", type=_rate, default=0.3, help=t("help.success_rate"))

    p = add("generate"); add_generate_args(p); add_cutoff(p)
    p = add("features"); add_cutoff(p)
    p = add("train"); p.add_argument("--seed", type=int, default=42, help=t("help.seed"))
    p = add("evaluate")
    p = add("threshold"); add_money(p)
    p.add_argument("--model", choices=("lr_weighted", "lr_plain", "rf"), default="lr_weighted", help=t("help.model"))
    p = add("score")
    p.add_argument("--top", type=_positive_int, default=1000, help=t("help.top"))
    p.add_argument("--model", choices=("lr_weighted", "lr_plain", "rf"), default="lr_weighted", help=t("help.model"))
    add("explain")
    p = add("all"); add_generate_args(p); add_cutoff(p); add_money(p)
    p.add_argument("--top", type=_positive_int, default=1000, help=t("help.top"))
    return parser


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(t("err.positive", value=text))
    return value


def _non_negative(text: str) -> float:
    value = float(text)
    if value < 0:
        raise argparse.ArgumentTypeError(t("err.non_negative", value=text))
    return value


def _rate(text: str) -> float:
    value = float(text)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError(t("err.rate", value=text))
    return value


def _iso_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(t("err.date", value=text)) from None


# ---------------------------------------------------------------- helpers

def _banner(key: str) -> None:
    print(f"\n=== {t(key)} ===")


def _cutoff(args, layout: Layout) -> date:
    if getattr(args, "cutoff", None):
        return args.cutoff
    meta = layout.read_meta()
    return date.fromisoformat(meta["cutoff"]) if "cutoff" in meta else DEFAULT_CUTOFF


def _need(path: str | Path, step: str) -> None:
    if not Path(str(path).replace("file:", "")).exists():
        print(t("err.missing_input", path=path, step=step), file=sys.stderr)
        raise SystemExit(1)


def _train_seed(layout: Layout) -> int:
    f = layout.model_dir / "train.json"
    return int(json.loads(f.read_text())["seed"]) if f.exists() else 42


def _test_pairs(spark, layout: Layout, model_name: str):
    """(probability, label) pairs for the held-out customers, using a saved model."""
    from .evaluate import collect_pairs
    from .train import load_model, split_train_test

    _need(layout.features, "features")
    _need(layout.model_path(model_name), "train")
    _, test = split_train_test(spark.read.parquet(layout.features), _train_seed(layout))
    return collect_pairs(load_model(layout.model_path(model_name)).transform(test))


# ---------------------------------------------------------------- steps

def step_generate(spark, args, layout: Layout) -> None:
    from .generate import make_frames

    _banner("step.generate")
    cutoff = _cutoff(args, layout)
    customers, events = make_frames(spark, args.customers, args.seed, cutoff)
    customers.write.mode("overwrite").parquet(layout.customers)
    events.write.mode("overwrite").parquet(layout.events)
    layout.write_meta({"customers": args.customers, "seed": args.seed, "cutoff": cutoff.isoformat()})
    n_events = spark.read.parquet(layout.events).count()
    print(t("generate.done", customers=args.customers, events=n_events, seed=args.seed, path=layout.data_dir))
    print(t("generate.fake"))


def step_features(spark, args, layout: Layout) -> None:
    from .features import build_features, summarize

    _banner("step.features")
    _need(layout.events, "generate")
    cutoff = _cutoff(args, layout)
    feats = build_features(spark.read.parquet(layout.customers), spark.read.parquet(layout.events), cutoff)
    feats.write.mode("overwrite").parquet(layout.features)
    s = summarize(spark.read.parquet(layout.features))
    print(t("features.windows", cutoff=cutoff, obs_start=cutoff - timedelta(days=120),
            label_end=cutoff + timedelta(days=LABEL_DAYS)))
    print(t("features.summary", customers=s["customers"], churned=s["churned"],
            rate=f"{s['churn_rate']:.1%}", zero=s["zero_activity"]))
    print(t("features.meaning", per100=round(s["churn_rate"] * 100)))


def step_train(spark, args, layout: Layout) -> None:
    from .train import fit_models, save_models, split_train_test

    _banner("step.train")
    _need(layout.features, "features")
    feats = spark.read.parquet(layout.features)
    train, test = split_train_test(feats, args.seed)
    train = train.cache()
    print(t("train.split", train=train.count(), test=test.count()))
    models = fit_models(train, args.seed)
    save_models(models, layout.model_path)
    layout.model_dir.mkdir(parents=True, exist_ok=True)
    (layout.model_dir / "train.json").write_text(json.dumps({"seed": args.seed}), encoding="utf-8")
    train.unpersist()
    print(t("train.done", names=", ".join(models), path=layout.model_dir))


def step_evaluate(spark, args, layout: Layout) -> None:
    from .evaluate import (auc_metrics, baseline_accuracy, collect_pairs, confusion_at, format_table,
                           lift_at, threshold_table)
    from .train import MODEL_NAMES, load_model, split_train_test

    _banner("step.evaluate")
    _need(layout.features, "features")
    for name in MODEL_NAMES:
        _need(layout.model_path(name), "train")
    _, test = split_train_test(spark.read.parquet(layout.features), _train_seed(layout))
    test = test.cache()
    results = {}
    for name in MODEL_NAMES:
        scored = load_model(layout.model_path(name)).transform(test).cache()
        pairs = collect_pairs(scored)
        c = confusion_at(pairs, 0.5)
        acc = (c["tp"] + c["tn"]) / len(pairs)
        results[name] = {**auc_metrics(scored), "pairs": pairs, "acc": acc, "recall": c["recall"],
                         "precision": c["precision"]}
        scored.unpersist()
    base_pairs = results["lr_weighted"]["pairs"]
    base_acc = baseline_accuracy(base_pairs)
    print(t("evaluate.trap", acc=f"{base_acc:.3f}", rate=f"{1 - base_acc:.3f}"))
    print(t("evaluate.models_header"))
    for name, r in results.items():
        print(f"  {name:<12} AUC-ROC {r['auc_roc']:.4f}   PR-AUC {r['pr_auc']:.4f}   "
              f"acc@0.5 {r['acc']:.3f}   precision@0.5 {r['precision']:.3f}   recall@0.5 {r['recall']:.3f}")
    print(t("evaluate.trap_meaning", acc=f"{base_acc:.3f}"))
    print(t("evaluate.weight_meaning", plain=f"{results['lr_plain']['recall']:.3f}",
            weighted=f"{results['lr_weighted']['recall']:.3f}"))
    print("\n" + t("evaluate.table_title", model="lr_weighted", n=len(base_pairs)))
    print(format_table(threshold_table(base_pairs), t("table.header")))
    lift = lift_at(base_pairs, 0.10)
    print("\n" + t("evaluate.lift", top=f"{lift['top_fraction']:.0%}", n=lift["n_top"],
                    top_rate=f"{lift['top_rate']:.1%}", base=f"{lift['base_rate']:.1%}", lift=f"{lift['lift']:.2f}"))
    print(t("evaluate.lift_meaning", lift=f"{lift['lift']:.1f}"))
    test.unpersist()


def step_threshold(spark, args, layout: Layout) -> None:
    from .evaluate import confusion_at
    from .threshold import best_threshold, profit, save_threshold

    _banner("step.threshold")
    pairs = _test_pairs(spark, layout, args.model)
    params = {"model": args.model, "contact_cost": args.contact_cost,
              "save_value": args.save_value, "success_rate": args.success_rate}
    best = best_threshold(pairs, args.contact_cost, args.save_value, args.success_rate)
    at_half = confusion_at(pairs, 0.5)
    half_profit = profit(at_half["tp"], at_half["fp"], args.contact_cost, args.save_value, args.success_rate)
    save_threshold(layout.threshold_file, best, params)
    print(t("threshold.formula", cost=args.contact_cost, value=args.save_value, rate=args.success_rate))
    print(t("threshold.best", threshold=f"{best['threshold']:.2f}", profit=f"{best['profit']:.1f}",
            contacted=best["contacted"], tp=best["tp"], fp=best["fp"], fn=best["fn"]))
    if best["profit"] <= 0:
        print(t("threshold.nowin"))
    print(t("threshold.vs_half", profit=f"{half_profit:.1f}"))
    print(t("threshold.saved", path=layout.threshold_file))
    print(t("threshold.meaning"))


def step_score(spark, args, layout: Layout) -> None:
    from pyspark.sql import functions as F

    from .features import build_features
    from .score import score_customers, top_rows, write_targets_csv
    from .threshold import load_threshold
    from .train import load_model

    _banner("step.score")
    _need(layout.events, "generate")
    _need(layout.model_path(args.model), "train")
    next_cutoff = _cutoff(args, layout) + timedelta(days=LABEL_DAYS)
    customers = spark.read.parquet(layout.customers)
    events = spark.read.parquet(layout.events)
    last_day = events.agg(F.max("event_date")).collect()[0][0]
    known = last_day is not None and last_day >= next_cutoff + timedelta(days=LABEL_DAYS - 1)
    feats = build_features(customers, events, next_cutoff, with_label=known)
    n_all = feats.count()
    # customers with no activity in the last 30 days have already left: nothing to save
    feats = feats.where(F.col("recency_days") <= ACTIVE_DAYS)
    print(t("score.active", active=feats.count(), total=n_all, days=ACTIVE_DAYS))
    scored = score_customers(load_model(layout.model_path(args.model)), feats).cache()
    threshold = load_threshold(layout.threshold_file)
    rows = top_rows(scored, args.top)
    path = write_targets_csv(rows, threshold, layout.targets_csv)
    n_above = sum(1 for _, p in rows if p >= threshold)
    print(t("score.done", n=len(rows), cutoff=next_cutoff, threshold=f"{threshold:.2f}", above=n_above, path=path))
    if known:
        ids = [cid for cid, _ in rows]
        labels = {int(r["customer_id"]): float(r["label"]) for r in scored.where(F.col("customer_id").isin(ids)).collect()}
        hit = sum(1 for cid in ids if labels[cid] == 1.0)
        overall = scored.where(F.col("label") == 1.0).count() / scored.count()
        print(t("score.check", n=len(ids), hit=hit, rate=f"{hit / len(ids):.1%}", overall=f"{overall:.1%}"))
    scored.unpersist()
    print(t("score.meaning"))


def step_explain(spark, args, layout: Layout) -> None:
    from .explain import lr_coefficients, rf_importances
    from .train import load_model

    _banner("step.explain")
    _need(layout.model_path("rf"), "train")
    print(t("explain.rf"))
    for name, value in rf_importances(load_model(layout.model_path("rf"))):
        print(f"  {name:<14}{value:7.3f}  {'#' * round(value * 40)}")
    print(t("explain.lr"))
    for name, value in lr_coefficients(load_model(layout.model_path("lr_weighted"))):
        print(f"  {name:<14}{value:+7.3f}")
    print(t("explain.meaning"))


STEPS = {
    "generate": step_generate, "features": step_features, "train": step_train,
    "evaluate": step_evaluate, "threshold": step_threshold, "score": step_score, "explain": step_explain,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    set_lang(_preparse_lang(argv))
    parser = build_parser()
    args = parser.parse_args(argv)
    set_lang(resolve_lang(getattr(args, "lang", None)))
    layout = Layout(args.data_dir, args.model_dir, args.out_dir)

    from .session import get_spark

    spark = get_spark()
    try:
        if args.command == "all":
            args.seed = getattr(args, "seed", 42)
            args.model = "lr_weighted"
            for name in ("generate", "features", "train", "evaluate", "threshold", "score", "explain"):
                STEPS[name](spark, args, layout)
        else:
            STEPS[args.command](spark, args, layout)
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
