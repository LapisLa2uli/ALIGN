"""Post-hoc interval-granularity sensitivity analysis of frozen real-test predictions.

Run with align-model/runs/env/bin/python. Does not alter source labels or predictions.
"""
from collections import defaultdict
import csv
import importlib.util
import json
import math
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
SOURCE = RUN / "test_results"
OUT = RUN / "interval_results"
SCORER = ROOT / "experiments/baselines_034_20260918_corrected/evaluate.py"
spec = importlib.util.spec_from_file_location("event_scorer", SCORER)
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)
GAPS = [None, 0, 50, 100, 200, 500]
CRITERIA = ["iou_0.3", "iou_0.5", "onset_50ms"]
MODELS = ["polytune", "laddersym"]


def merge(labels, gap_ms):
    """Union same-type intervals transitively; fill gaps <= gap_ms. Keep lineage."""
    groups = defaultdict(list)
    for i, label in enumerate(labels):
        start, end = float(label["start_time"]), float(label["end_time"])
        assert math.isfinite(start) and math.isfinite(end) and 0 <= start < end
        groups[label["type"]].append(dict(type=label["type"], start_time=start,
                                          end_time=end, source_indices=[i]))
    merged = []
    for kind, rows in sorted(groups.items()):
        current = []
        for row in sorted(rows, key=lambda r: (r["start_time"], r["end_time"])):
            # Preserve unclassified decoder outputs as separate false positives.
            if (gap_ms is not None and kind != "unclassified" and current
                    and (gap_ms == "all" or
                         row["start_time"] <= current[-1]["end_time"] + gap_ms / 1000 + 1e-9)):
                current[-1]["end_time"] = max(current[-1]["end_time"], row["end_time"])
                current[-1]["source_indices"].extend(row["source_indices"])
            else:
                current.append(row)
        merged.extend(current)
    return sorted(merged, key=lambda r: (r["start_time"], r["end_time"], r["type"]))


def setting_name(gap):
    return "unmerged" if gap is None else "all_same_type" if gap == "all" else f"gap_{gap}ms"


def checks():
    def span(s, e, t="extra_note"):
        return dict(type=t, start_time=s, end_time=e)
    # Nested intervals must not shorten the union; a different type must stay separate.
    rows = [span(0, 1), span(.2, .3), span(1, 1.1), span(.5, .6, "wrong_note")]
    m = merge(rows, 0)
    assert len(m) == 2 and m[0]["end_time"] == 1.1 and len(m[0]["source_indices"]) == 3
    assert len(merge([span(0, 1), span(1.05, 2)], 50)) == 1
    assert len(merge([span(0, 1), span(1.051, 2)], 50)) == 2
    assert len(merge(rows, None)) == 4
    assert len(merge([span(0, 1, "unclassified")] * 2, 500)) == 2
    assert merge([], 100) == []
    assert len(merge([span(0, 1), span(100, 101), span(2, 3, "wrong_note")], "all")) == 2
    assert len(merge([span(0, 1, "unclassified")] * 2, "all")) == 2
    # Exclusive matching: two predictions cannot both get credit for one target.
    assert scorer.counts([span(0, 1)], [span(0, 1)] * 2, "iou_0.3")["matched"] == 1
    assert scorer.counts([span(0, 1)], [span(0, 1, "wrong_note")], "iou_0.3")["matched"] == 0


def main():
    checks()
    OUT.mkdir(exist_ok=True)
    dataset = scorer.read(SOURCE / "dataset_manifest.json")
    original = scorer.read(SOURCE / "results.json")
    raw_gold = {s: scorer.read(SOURCE / "bundles" / s / "labels.json")["labels"]
                for s in dataset["all_ids"]}
    gold = {s: [l for l in rows if l["type"] in scorer.FIVE_TYPES] for s, rows in raw_gold.items()}
    gold_hashes = {s: scorer.sha(SOURCE / "bundles" / s / "labels.json") for s in gold}
    assert gold_hashes == original["gold_sha256"]
    protocol = dict(status="post-hoc sensitivity analysis; no test-selected primary setting",
                    prediction_source=str(SOURCE), gaps_ms=GAPS, criteria=CRITERIA,
                    merging="prediction only, same type, transitive interval union plus gap filling; unclassified not merged",
                    gold="original labels unchanged, including duplicates; five task types",
                    matching="same type, maximum-cardinality one-to-one; pooled micro",
                    clean_clips="included; every unmatched prediction is a false positive",
                    taxonomy=scorer.FIVE_TYPES, dataset=dataset, gold_sha256=gold_hashes,
                    code_sha256={str(p): scorer.sha(p) for p in [Path(__file__), SCORER]},
                    limitation="Merging can join distinct errors or bridge correct notes; no duration cap or GT-dependent splitting. This is not the paper canonical F1.")
    scorer.write(OUT / "protocol.json", protocol)
    result = dict(protocol=protocol, models={})
    summary = []
    for model in MODELS:
        folder = SOURCE / (model + "_frozen")
        manifest = scorer.read(folder / "freeze_manifest.json")
        predictions = {}
        for row in manifest["rows"]:
            path = folder / (row["sample"] + ".json")
            assert scorer.sha(path) == row["output_sha256"]
            predictions[row["sample"]] = [l for l in scorer.read(path)["labels"]
                                            if l["type"] in scorer.FIVE_TYPES or l["type"] == "unclassified"]
        assert set(predictions) == set(gold)
        model_report = dict(freeze_manifest_sha256=scorer.sha(folder / "freeze_manifest.json"), settings={})
        result["models"][model] = model_report
        for gap in GAPS:
            name = setting_name(gap)
            pred = {s: merge(rows, gap) for s, rows in predictions.items()}
            scorer.write(OUT / f"{model}_{name}_spans.json", pred)
            settings = {}
            model_report["settings"][name] = settings
            for subset, key in [("full40", "all_ids"), ("filtered30", "filtered_ids")]:
                ids = dataset[key]
                empty = [s for s in ids if not raw_gold[s]]
                metrics = {}
                settings[subset] = dict(metrics=metrics, empty_gold_clips=empty,
                    predictions_on_empty_gold=sum(len(pred[s]) for s in empty),
                    per_clip_prediction_counts={s: len(pred[s]) for s in ids})
                for criterion in CRITERIA:
                    rows = {s: scorer.counts(gold[s], pred[s], criterion) for s in ids}
                    micro = scorer.pool(list(rows.values()))
                    if gap is None:
                        expected = original["models"][model]["subsets"][subset]["results"]["five_type"][criterion]["micro"]
                        assert micro == expected, (model, subset, criterion, micro, expected)
                    per_type = {t: scorer.pool([scorer.counts([l for l in gold[s] if l["type"] == t],
                                [l for l in pred[s] if l["type"] == t], criterion) for s in ids])
                                for t in scorer.FIVE_TYPES}
                    boot = scorer.bootstrap(list(rows.values()))
                    metrics[criterion] = dict(micro=micro, bootstrap=boot, per_clip=rows, per_type=per_type)
                    summary.append(dict(model=model, subset=subset, setting=name, criterion=criterion,
                                        **micro, ci_low=boot["lower_95"], ci_high=boot["upper_95"],
                                        predictions_on_empty_gold=settings[subset]["predictions_on_empty_gold"]))
            print(model, name, settings["full40"]["metrics"]["iou_0.3"]["micro"], flush=True)
    scorer.write(OUT / "results.json", result)
    with (OUT / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = ["# 区间级 F1：预测合并敏感性分析", "",
             "使用重训后的两个模型和修正后的 40 条测试录音；filtered30 沿用指定排除名单。",
             "同类预测区间相交、相接或间隔不超过给定阈值时，递归合并为覆盖整个范围的区间。不同类型不合并。",
             "人工 GT 保持原样（包括重复标签）；五类错误、同类型、最大数量一对一匹配、micro 汇总。19 条完全无错误录音仍计入全量。",
             "这是事后敏感性分析，所有预设间隔均报告，没有从测试集选定正式阈值。0ms 表示合并相交/相接区间，和不合并不同。",
             "如包含‘同类全部合并’，表示每条录音内每种错误类型的所有预测合成一个首尾覆盖区间；不跨录音、不合并不同类型，unclassified 保持单独计数。",
             "合并可能连接独立错误、跨过正确音符并使区间过长；IoU 同时约束预测与 GT 的覆盖范围。", ""]
    for subset in ["full40", "filtered30"]:
        lines += [f"## {subset}", "", "| 模型 | 合并间隔 | 预测区间数 | GT 数 | 命中 IoU≥0.3 | P | R | F1 IoU≥0.3 | F1 IoU≥0.5 | 无错误录音上的预测数 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for model in MODELS:
            for gap in GAPS:
                name = setting_name(gap)
                sub = result["models"][model]["settings"][name][subset]
                m = sub["metrics"]["iou_0.3"]["micro"]
                f5 = sub["metrics"]["iou_0.5"]["micro"]["f1"]
                label = "不合并" if gap is None else "同类全部合并" if gap == "all" else f"{gap}ms"
                lines.append(f"| {model} | {label} | {m['predicted']} | {m['gold']} | {m['matched']} | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {f5:.6f} | {sub['predictions_on_empty_gold']} |")
        lines.append("")
    lines += ["## 复核与文件", "", "- 不合并时的三个计分规则、两个模型、两个子集，全部精确复现之前保存的结果。",
              "- GT 和原始预测逐文件 SHA256 通过；嵌套区间、阈值边界、类型隔离与一对一匹配检查通过。",
              "- results.json：逐类型、逐录音结果及 2000 次录音级 bootstrap 95% 区间；summary.csv：汇总。",
              "- *_spans.json：各设置下的预测区间和原始预测索引，便于复查合并过程。",
              "- onset_50ms 仅用于对照，主要区间指标为 IoU。正常音符不计入，不能与论文 canonical F1 直接比较。"]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(OUT / "RESULTS.md")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extended", action="store_true", help="Evaluate 1/2/5/10-second and all-same-type merges in a separate output directory.")
    if parser.parse_args().extended:
        GAPS = [None, 500, 1000, 2000, 5000, 10000, "all"]
        OUT = RUN / "interval_results_extended"
    main()
