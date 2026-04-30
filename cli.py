import argparse
import datetime
import os
import platform
import importlib
import pkgutil
import inspect
import sys

from adapters import get_adapter
from schema import BenchmarkReport, TestCaseResult

BACKENDS = ["torch", "torch-npu", "torch4ms", "mindtorch", "mindnlp_patch"]

def load_suites():
    import suites
    suite_modules = []
    for _, module_name, _ in pkgutil.iter_modules(suites.__path__):
        module = importlib.import_module(f"suites.{module_name}")
        suite_modules.append(module)
    return suite_modules

def _split_filters(values):
    if not values:
        return set()
    selected = set()
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                selected.add(item)
    return selected

def _iter_suite_tests(suite_module):
    for name, obj in inspect.getmembers(suite_module):
        if inspect.isfunction(obj) and name.startswith("test_"):
            yield name, obj

def _matches_test_filter(suite_name, test_name, test_filters):
    if not test_filters:
        return True
    return (
        test_name in test_filters
        or f"{suite_name}/{test_name}" in test_filters
        or f"{suite_name}::{test_name}" in test_filters
    )

def select_suites(suite_modules, suite_filters=None, layer_filters=None, test_filters=None):
    suite_filters = suite_filters or set()
    layer_filters = layer_filters or set()
    test_filters = test_filters or set()
    selected = []
    for suite_module in suite_modules:
        layer = getattr(suite_module, "LAYER", "unknown")
        suite_name = suite_module.__name__.split(".")[-1]
        if suite_filters and suite_name not in suite_filters and suite_module.__name__ not in suite_filters:
            continue
        if layer_filters and layer not in layer_filters:
            continue
        tests = [
            (name, obj)
            for name, obj in _iter_suite_tests(suite_module)
            if _matches_test_filter(suite_name, name, test_filters)
        ]
        if tests:
            selected.append((suite_module, tests))
    return selected

def run_tests_in_suite(adapter, suite_module, tests=None):
    results = []
    layer = getattr(suite_module, "LAYER", "unknown")
    suite_name = suite_module.__name__.split(".")[-1]
    tests = list(tests) if tests is not None else list(_iter_suite_tests(suite_module))

    for name, obj in tests:
        print(f"Running {name} in {suite_name}...")
        try:
            raw_result = obj(adapter)
            if raw_result is None:
                continue

            if isinstance(raw_result, dict):
                skipped = bool(raw_result.get("skipped", False))
                compatibility = bool(raw_result.get("compatibility", True if skipped else False))
                correctness = raw_result.get("correctness", None if skipped else compatibility)
                performance_ms = raw_result.get("performance_ms", getattr(adapter, "last_performance_ms", 0.0))
                error_message = raw_result.get("error_message")
                usability_score = raw_result.get("usability_score")
            else:
                skipped = False
                success = bool(raw_result)
                compatibility = success
                correctness = success
                performance_ms = getattr(adapter, "last_performance_ms", 0.0)
                error_message = None
                usability_score = None

            results.append(TestCaseResult(
                test_name=name,
                suite_name=suite_name,
                layer=layer,
                compatibility=compatibility,
                correctness=correctness,
                performance_ms=performance_ms,
                error_message=error_message,
                usability_score=usability_score,
                skipped=skipped,
            ))

            # Clear metric for next test
            if hasattr(adapter, "last_performance_ms"):
                delattr(adapter, "last_performance_ms")

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
                        choices=BACKENDS,
                        help="Target backend to benchmark")
    parser.add_argument("--output", type=str, default="artifacts/reports/report.json",
                        help="Output JSON report file path")
    parser.add_argument("--suite", action="append",
                        help="Run only selected suite(s), e.g. module_smoke. Can be repeated or comma-separated.")
    parser.add_argument("--test", action="append",
                        help="Run only selected test(s), e.g. test_batchnorm2d_module or module_smoke/test_batchnorm2d_module. Can be repeated or comma-separated.")
    parser.add_argument("--layer", action="append",
                        help="Run only selected layer(s), e.g. operator, module, model, autograd, end2end. Can be repeated or comma-separated.")
    args = parser.parse_args()

    print(f"Starting Benchmark for backend: {args.backend}")
    suites = load_suites()
    suite_filters = _split_filters(args.suite)
    test_filters = _split_filters(args.test)
    layer_filters = _split_filters(args.layer)
    selected_suites = select_suites(
        suites,
        suite_filters=suite_filters,
        layer_filters=layer_filters,
        test_filters=test_filters,
    )
    if not selected_suites:
        available = [
            f"{suite.__name__.split('.')[-1]}/{test_name}"
            for suite in suites
            for test_name, _ in _iter_suite_tests(suite)
        ]
        raise SystemExit(
            "No tests matched selection. "
            f"suite={sorted(suite_filters) or 'ALL'}, "
            f"test={sorted(test_filters) or 'ALL'}, "
            f"layer={sorted(layer_filters) or 'ALL'}.\n"
            "Available tests:\n  " + "\n  ".join(sorted(available))
        )
    selected_count = sum(len(tests) for _, tests in selected_suites)
    print(f"Selected {selected_count} test(s) from {len(selected_suites)} suite(s).")

    adapter = get_adapter(args.backend)
    
    setup_failed = False
    setup_error = None
    try:
        adapter.setup()
    except Exception as e:
        setup_failed = True
        setup_error = str(e)
        print(f"Adapter setup failed: {e}")

    all_results = []
    
    if setup_failed:
        for suite, tests in selected_suites:
            layer = getattr(suite, "LAYER", "unknown")
            suite_name = suite.__name__.split(".")[-1]
            for name, _ in tests:
                all_results.append(TestCaseResult(
                    test_name=name,
                    suite_name=suite_name,
                    layer=layer,
                    compatibility=False,
                    correctness=False,
                    error_message=f"Setup failed: {setup_error}"
                ))
    else:
        for suite, tests in selected_suites:
            all_results.extend(run_tests_in_suite(adapter, suite, tests=tests))
        try:
            adapter.teardown()
        except Exception as e:
            print(f"Adapter teardown failed: {e}")

    # Collect Environment Info
    env_info = {
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "os": platform.system()
    }
    for module_name in ("torch", "torch_npu", "mindspore", "torch4ms", "mindtorch_v2", "mindnlp", "torchvision", "ultralytics"):
        try:
            module = importlib.import_module(module_name)
            env_info[module_name] = getattr(module, "__version__", "present")
        except Exception as exc:
            env_info[module_name] = f"unavailable: {exc}"

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
