import argparse
import glob
import json
import os
from collections import defaultdict
from datetime import datetime


def _is_success(res):
    if res.get("skipped", False):
        return False
    if not res.get("compatibility", False):
        return False
    correctness = res.get("correctness")
    return True if correctness is None else bool(correctness)


def _active_results(results):
    return [res for res in results if not res.get("skipped", False)]


def _load_reports(input_glob, latest_per_backend=True):
    report_files = glob.glob(input_glob)
    if not report_files:
        return [], []

    report_files = sorted(report_files, key=os.path.getmtime, reverse=True)
    loaded = []
    for report_file in report_files:
        with open(report_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["_report_file"] = os.path.basename(report_file)
        loaded.append(data)

    if not latest_per_backend:
        return loaded, report_files

    picked = {}
    for data in loaded:
        backend = data.get("backend")
        if backend and backend not in picked:
            picked[backend] = data
    return list(picked.values()), report_files


def _format_status(res):
    if res is None:
        return "N/A"
    if res.get("skipped", False):
        return "SKIP"
    if not res.get("compatibility", False):
        return "FAIL"
    correctness = res.get("correctness")
    if correctness is False:
        return "WARN"
    perf = res.get("performance_ms")
    if perf is not None and perf > 0:
        return f"PASS ({perf:.1f}ms)"
    return "PASS"


def generate_markdown_report(output_path="benchmark_report.md", input_glob="report_*.json", latest_per_backend=True):
    data_all, scanned_reports = _load_reports(input_glob, latest_per_backend=latest_per_backend)
    if not data_all:
        print(f"No JSON reports found by pattern: {input_glob}")
        return

    all_tests = set()
    for data in data_all:
        for res in data["results"]:
            all_tests.add((res["suite_name"], res["test_name"], res.get("layer", "unknown")))
    all_tests = sorted(list(all_tests))

    backends = sorted(d["backend"] for d in data_all)
    data_map = {d["backend"]: d for d in data_all}
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = []
    md.append("# TorchBridge Benchmark Report")
    md.append("")
    md.append(f"**Generated at:** {now_str}")
    md.append(f"**Input pattern:** `{input_glob}`")
    md.append(f"**Selection mode:** `{'latest_per_backend' if latest_per_backend else 'all_reports'}`")
    md.append(f"**Scanned reports:** `{len(scanned_reports)}`")
    md.append("")

    md.append("## 1. Overall Summary")
    md.append("")
    md.append("| Backend | Compat Pass Rate | Full Pass Rate | Passed (Full) | Skipped | Perf Coverage | Avg Perf(ms) | Report File |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for backend in backends:
        data = data_map[backend]
        results = data["results"]
        active = _active_results(results)
        total = len(active)
        skipped = len(results) - total
        compat_pass = sum(1 for r in active if r.get("compatibility", False))
        full_pass = sum(1 for r in active if _is_success(r))
        perf_vals = [float(r["performance_ms"]) for r in active if r.get("performance_ms") not in (None, 0)]
        compat_rate = (compat_pass / total * 100.0) if total else 0.0
        full_rate = (full_pass / total * 100.0) if total else 0.0
        perf_cov = (len(perf_vals) / total * 100.0) if total else 0.0
        avg_perf = (sum(perf_vals) / len(perf_vals)) if perf_vals else 0.0
        md.append(
            f"| `{backend}` | {compat_rate:.1f}% | {full_rate:.1f}% | {full_pass} / {total} | "
            f"{skipped} | {perf_cov:.1f}% | {avg_perf:.1f} | `{data.get('_report_file', 'N/A')}` |"
        )
    md.append("")

    md.append("## 2. Suite Breakdown")
    md.append("")
    md.append("| Backend | Suite | Layer | Full Pass | Passed / Total | Skipped | Avg Perf(ms) |")
    md.append("|---|---|---|---:|---:|---:|---:|")
    for backend in backends:
        results = data_map[backend]["results"]
        by_suite = defaultdict(list)
        for r in results:
            by_suite[(r["suite_name"], r.get("layer", "unknown"))].append(r)
        for (suite_name, layer), rows in sorted(by_suite.items()):
            active = _active_results(rows)
            total = len(active)
            skipped = len(rows) - total
            passed = sum(1 for r in active if _is_success(r))
            pass_rate = (passed / total * 100.0) if total else 0.0
            perf_vals = [float(r["performance_ms"]) for r in active if r.get("performance_ms") not in (None, 0)]
            avg_perf = (sum(perf_vals) / len(perf_vals)) if perf_vals else 0.0
            md.append(f"| `{backend}` | `{suite_name}` | `{layer}` | {pass_rate:.1f}% | {passed} / {total} | {skipped} | {avg_perf:.1f} |")
    md.append("")

    md.append("## 3. Layer Breakdown")
    md.append("")
    md.append("| Backend | Layer | Full Pass | Passed / Total |")
    md.append("|---|---|---:|---:|")
    for backend in backends:
        results = data_map[backend]["results"]
        by_layer = defaultdict(list)
        for r in results:
            by_layer[r.get("layer", "unknown")].append(r)
        for layer, rows in sorted(by_layer.items()):
            active = _active_results(rows)
            total = len(active)
            passed = sum(1 for r in active if _is_success(r))
            pass_rate = (passed / total * 100.0) if total else 0.0
            md.append(f"| `{backend}` | `{layer}` | {pass_rate:.1f}% | {passed} / {total} |")
    md.append("")

    md.append("## 4. Detailed Matrix")
    md.append("")
    md.append("| Suite | Layer | Test Name | " + " | ".join([f"`{b}` (Status / Time)" for b in backends]) + " |")
    md.append("|---|---|---|" + "|".join(["---" for _ in backends]) + "|")
    for suite, test, layer in all_tests:
        row = f"| `{suite}` | `{layer}` | `{test}` |"
        for b in backends:
            b_data = data_map[b]
            res = next((r for r in b_data["results"] if r["test_name"] == test and r["suite_name"] == suite), None)
            row += f" {_format_status(res)} |"
        md.append(row)
    md.append("")

    md.append("## 5. Slowest Tests")
    md.append("")
    for backend in backends:
        perf_rows = [r for r in data_map[backend]["results"] if not r.get("skipped", False) and r.get("performance_ms") not in (None, 0)]
        perf_rows.sort(key=lambda r: float(r["performance_ms"]), reverse=True)
        md.append(f"### Backend: `{backend}`")
        if not perf_rows:
            md.append("- No performance records.")
            md.append("")
            continue
        for r in perf_rows[:5]:
            md.append(f"- `{r['suite_name']}::{r['test_name']}`: {float(r['performance_ms']):.1f} ms")
        md.append("")

    md.append("## 6. Failure Diagnostics")
    md.append("")
    has_failures = False
    for backend in backends:
        failures = [r for r in data_map[backend]["results"] if not r.get("skipped", False) and not _is_success(r)]
        if not failures:
            continue
        has_failures = True
        md.append(f"### Backend: `{backend}`")
        grouped = defaultdict(list)
        for f in failures:
            grouped[f["suite_name"]].append(f)
        for suite_name, rows in sorted(grouped.items()):
            md.append(f"- Suite `{suite_name}`: {len(rows)} failure(s)")
            for f in rows:
                err = str(f.get("error_message", "Unknown Error")).replace("\n", "<br>")
                md.append(f"  - `{f['test_name']}`: `{err}`")
        md.append("")
    if not has_failures:
        md.append("All tests passed for all selected backends.")
        md.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"Markdown report generated: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate markdown report from benchmark JSON files.")
    parser.add_argument("--output", default="benchmark_report.md", help="Output markdown file path.")
    parser.add_argument("--input-glob", default="report_*.json", help="Input JSON glob pattern.")
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Use all matched reports instead of latest report per backend.",
    )
    args = parser.parse_args()
    generate_markdown_report(output_path=args.output, input_glob=args.input_glob, latest_per_backend=not args.all_runs)
