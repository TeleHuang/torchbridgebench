import argparse
import datetime
import os
import platform
import importlib
import pkgutil
import inspect

from adapters import get_adapter
from schema import BenchmarkReport, TestCaseResult

def load_suites():
    import suites
    suite_modules = []
    for _, module_name, _ in pkgutil.iter_modules(suites.__path__):
        module = importlib.import_module(f"suites.{module_name}")
        suite_modules.append(module)
    return suite_modules

def run_tests_in_suite(adapter, suite_module):
    results = []
    layer = getattr(suite_module, "LAYER", "unknown")
    suite_name = suite_module.__name__.split(".")[-1]
    
    for name, obj in inspect.getmembers(suite_module):
        if inspect.isfunction(obj) and name.startswith("test_"):
            print(f"Running {name} in {suite_name}...")
            try:
                # Mock result for smoke test execution
                # A real implementation would capture execution time, output, and correctness
                success = obj(adapter)
                results.append(TestCaseResult(
                    test_name=name,
                    suite_name=suite_name,
                    layer=layer,
                    compatibility=success,
                    correctness=success,
                    performance_ms=0.0
                ))
            except Exception as e:
                results.append(TestCaseResult(
                    test_name=name,
                    suite_name=suite_name,
                    layer=layer,
                    compatibility=False,
                    correctness=False,
                    error_message=str(e)
                ))
    return results

def main():
    parser = argparse.ArgumentParser(description="TorchBridgeBench Runner")
    parser.add_argument("--backend", type=str, required=True, 
                        choices=["torch", "torch-npu", "torch4ms", "mindtorch", "mindnlp_patch"],
                        help="Target backend to benchmark")
    parser.add_argument("--output", type=str, default="report.json",
                        help="Output JSON report file path")
    args = parser.parse_args()

    print(f"Starting Benchmark for backend: {args.backend}")
    adapter = get_adapter(args.backend)
    adapter.setup()

    all_results = []
    suites = load_suites()
    for suite in suites:
        all_results.extend(run_tests_in_suite(adapter, suite))

    adapter.teardown()

    # Collect Environment Info
    env_info = {
        "python": platform.python_version(),
        "os": platform.system()
    }

    report = BenchmarkReport(
        backend=args.backend,
        timestamp=datetime.datetime.now().isoformat(),
        environment=env_info,
        results=all_results
    )

    report.save(args.output)
    print(f"Benchmark finished. Report saved to {args.output}")

if __name__ == "__main__":
    main()
