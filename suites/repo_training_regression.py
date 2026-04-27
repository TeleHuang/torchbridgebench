import os
import subprocess
import sys
import time
from pathlib import Path


LAYER = "repo_regression"

_ASCEND_TORCH4MS_ROOT = Path(
    os.environ.get("TORCH4MS_REPO_ROOT", "/root/autodl-tmp/ascend-torch4ms-ms272-stable")
)


def _skip_result(reason: str):
    return {
        "compatibility": True,
        "correctness": None,
        "performance_ms": 0.0,
        "error_message": reason,
        "skipped": True,
    }


def _module_available(module_name: str) -> bool:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)",
            module_name,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return proc.returncode == 0


def _tail(text: str, lines: int = 40) -> str:
    rows = [line for line in (text or "").splitlines() if line.strip()]
    if not rows:
        return "No output captured."
    return "\n".join(rows[-lines:])


def _run_repo_script(
    adapter,
    script_relpath: str,
    *script_args: str,
    required_markers: tuple[str, ...] = (),
    forbidden_markers: tuple[str, ...] = (),
    timeout_sec: int = 1800,
):
    if getattr(adapter, "name", "") != "torch4ms":
        return None

    start = time.time()
    env = os.environ.copy()
    env.setdefault("TORCHDYNAMO_DISABLE", "1")
    cmd = [sys.executable, str(_ASCEND_TORCH4MS_ROOT / script_relpath), *map(str, script_args)]
    proc = subprocess.run(
        cmd,
        cwd=_ASCEND_TORCH4MS_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_sec,
    )
    adapter.last_performance_ms = (time.time() - start) * 1000.0

    output = proc.stdout or ""
    if proc.returncode != 0:
        raise RuntimeError(_tail(output))

    for marker in forbidden_markers:
        if marker in output:
            raise RuntimeError(f"Detected forbidden marker `{marker}`.\n{_tail(output)}")

    missing = [marker for marker in required_markers if marker not in output]
    if missing:
        raise RuntimeError(f"Missing required marker(s): {missing}\n{_tail(output)}")

    return True


def test_repo_backward_script(adapter):
    return _run_repo_script(
        adapter,
        "test_backward_benchmark_style.py",
        required_markers=("OK: torch4ms PyTorch-style backward matches torch",),
    )


def test_repo_train_cnn_script(adapter):
    return _run_repo_script(
        adapter,
        "test_train_cnn.py",
        required_markers=("=== stability regression summary ===",),
        forbidden_markers=("BAD_INVALID", "MISMATCH_ZERO_GRAD"),
    )


def test_repo_train_transformer_script(adapter):
    return _run_repo_script(
        adapter,
        "test_train_transformer.py",
        required_markers=("=== transformer stability summary ===",),
        forbidden_markers=("BAD_INVALID", "MISMATCH_ZERO_GRAD"),
    )


def test_repo_train_resnet_compare_script(adapter):
    return _run_repo_script(
        adapter,
        "test_train_resnet_compare.py",
        "--use-fake-data",
        "--device-target",
        "CPU",
        "--batch-size",
        "8",
        forbidden_markers=("BAD_INVALID", "MISMATCH_ZERO"),
    )


def test_experiment_resnet_cifar_torch4ms(adapter):
    return _run_repo_script(
        adapter,
        "experiments/resnet_torchax_cifar/train_resnet_cifar_torchax.py",
        "--device",
        "cpu",
        "--train-steps",
        "2",
        "--val-steps",
        "1",
        "--use-fake-data",
        required_markers=("[result] RESNET_CIFAR_TORCHAX_SMOKE=PASS",),
    )


def test_experiment_mobilenet_torch4ms(adapter):
    return _run_repo_script(
        adapter,
        "experiments/mobilenet_torchax_cifar/run_mobilenet_experiment.py",
        "--mode",
        "torch4ms",
        "--train-steps",
        "2",
        "--val-steps",
        "1",
        "--use-fake-data",
        "--output",
        "artifacts/benchmark_mobilenet_experiment_report.json",
        required_markers=("[torch4ms][summary]",),
    )


def test_experiment_yolo_ultralytics_torch4ms(adapter):
    if getattr(adapter, "name", "") == "torch4ms" and not _module_available("ultralytics"):
        return _skip_result("ultralytics is not installed in the torch4ms benchmark interpreter")

    return _run_repo_script(
        adapter,
        "experiments/yolo_ultralytics_smoke/run_yolo_ultralytics_smoke.py",
        "--mode",
        "torch4ms",
        "--epochs",
        "1",
        "--batch",
        "2",
        "--imgsz",
        "64",
        "--num-train",
        "4",
        "--num-val",
        "2",
        "--project",
        "artifacts/yolo_runs_benchmark",
        "--run-name",
        "yolo_smoke_benchmark",
        "--output",
        "artifacts/yolo_benchmark_report.json",
        required_markers=("[result] mode=torch4ms, status=success",),
    )
