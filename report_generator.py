import json
import glob
from datetime import datetime
import os

def generate_markdown_report():
    reports = glob.glob("report_*.json")
    if not reports:
        print("No JSON reports found.")
        return

    # Keep only the newest report per backend to avoid duplicate backend columns.
    reports = sorted(reports, key=os.path.getmtime, reverse=True)
    data_by_backend = {}
    for report_file in reports:
        with open(report_file, 'r') as f:
            data = json.load(f)
        backend = data.get("backend")
        if backend and backend not in data_by_backend:
            data_by_backend[backend] = data

    data_all = list(data_by_backend.values())
    all_tests = set()
    for data in data_all:
        for res in data['results']:
            all_tests.add((res['suite_name'], res['test_name']))

    all_tests = sorted(list(all_tests))
    
    md_content = f"# TorchBridge Benchmark Report\n\n"
    md_content += f"**Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    md_content += "## 1. Summary\n\n"
    md_content += "| Backend | Pass Rate | Passed / Total |\n"
    md_content += "|---------|-----------|----------------|\n"
    
    for data in sorted(data_all, key=lambda x: x['backend']):
        passed = sum(1 for r in data['results'] if r['compatibility'])
        total = len(data['results'])
        pass_rate = (passed / total) * 100 if total > 0 else 0
        md_content += f"| `{data['backend']}` | {pass_rate:.1f}% | {passed} / {total} |\n"
        
    md_content += "\n## 2. Detailed Test Results\n\n"
    
    backends = sorted([d['backend'] for d in data_all])
    
    # Table Header (Adding Performance Metric support)
    md_content += "| Suite | Test Name | " + " | ".join([f"`{b}` (Status / Time)" for b in backends]) + " |\n"
    md_content += "|---|---|" + "|".join(["---" for _ in backends]) + "|\n"
    
    for suite, test in all_tests:
        row = f"| {suite} | {test} |"
        for b in backends:
            b_data = next((d for d in data_all if d['backend'] == b), None)
            if not b_data:
                row += " N/A |"
                continue
                
            res = next((r for r in b_data['results'] if r['test_name'] == test and r['suite_name'] == suite), None)
            
            if res is None:
                row += " N/A |"
            elif res['compatibility']:
                perf = res.get('performance_ms', 0.0)
                if perf and perf > 0:
                    row += f" ✅ ({perf:.1f}ms) |"
                else:
                    row += " ✅ |"
            else:
                row += " ❌ |"
        md_content += row + "\n"
        
    md_content += "\n## 3. Failure Diagnostics\n\n"
    
    has_failures = False
    for data in sorted(data_all, key=lambda x: x['backend']):
        failures = [r for r in data['results'] if not r['compatibility']]
        if failures:
            has_failures = True
            md_content += f"### Backend: `{data['backend']}`\n\n"
            for f in failures:
                err = str(f.get('error_message', 'Unknown Error')).replace('\n', '<br>')
                md_content += f"- **{f['suite_name']}::{f['test_name']}**: `{err}`\n"
            md_content += "\n"
            
    if not has_failures:
        md_content += "All tests passed for all backends! 🎉\n"
        
    with open("benchmark_report.md", "w") as f:
        f.write(md_content)
        
    print(f"Markdown report generated: {os.path.abspath('benchmark_report.md')}")

if __name__ == "__main__":
    generate_markdown_report()
