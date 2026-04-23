import json
import glob

reports = glob.glob("report_*.json")
for report in reports:
    with open(report, "r") as f:
        data = json.load(f)
        backend = data["backend"]
        results = data["results"]
        passed = sum(1 for r in results if r["compatibility"])
        total = len(results)
        print(f"Backend: {backend:<15} | Score: {passed}/{total} passed ({(passed/total)*100:.1f}%)")
