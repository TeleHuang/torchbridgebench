import json
import glob

reports = glob.glob("report_*.json")
for report in reports:
    with open(report, "r") as f:
        data = json.load(f)
        backend = data["backend"]
        results = [r for r in data["results"] if not r.get("skipped", False)]
        skipped = len(data["results"]) - len(results)
        passed = sum(1 for r in results if r["compatibility"])
        total = len(results)
        rate = (passed / total) * 100 if total else 0.0
        print(f"Backend: {backend:<15} | Score: {passed}/{total} passed ({rate:.1f}%), skipped={skipped}")
