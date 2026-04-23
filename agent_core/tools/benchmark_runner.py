import os
import subprocess
import json

class BenchmarkRunnerTool:
    """Tool for running specific parts of the benchmark suite safely."""
    
    def __init__(self, workspace_path):
        self.workspace_path = workspace_path
        self.cli_path = os.path.join(workspace_path, "cli.py")
        
        if not os.path.exists(self.cli_path):
            raise FileNotFoundError(f"Benchmark CLI not found at {self.cli_path}")

    def run_suite(self, venv_python_path, backend, suite_name, output_file="agent_temp_report.json"):
        """
        Runs a specific test suite (e.g., 'operator_smoke') for a specific backend
        using the specified virtual environment's Python executable.
        """
        cmd = [
            venv_python_path,
            self.cli_path,
            "--backend", backend,
            "--output", output_file,
            # In a real implementation, cli.py would need an update to accept --suite
            # For now, we simulate the capability or assume cli.py has been modified
            "--suite", suite_name
        ]
        
        result = subprocess.run(cmd, cwd=self.workspace_path, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(os.path.join(self.workspace_path, output_file)):
            with open(os.path.join(self.workspace_path, output_file), 'r') as f:
                report_data = json.load(f)
            return {
                "success": True,
                "stdout": result.stdout,
                "report": report_data
            }
        else:
            return {
                "success": False,
                "error": "Benchmark execution failed or report not generated.",
                "stdout": result.stdout,
                "stderr": result.stderr
            }

    def run_single_test(self, venv_python_path, backend, suite_name, test_name, output_file="agent_temp_report.json"):
        """Runs a single test case to quickly verify a fix."""
        cmd = [
            venv_python_path,
            self.cli_path,
            "--backend", backend,
            "--output", output_file,
            "--suite", suite_name,
            "--test", test_name
        ]
        
        result = subprocess.run(cmd, cwd=self.workspace_path, capture_output=True, text=True)
        # Similar logic to parse report
        if result.returncode == 0 and os.path.exists(os.path.join(self.workspace_path, output_file)):
            with open(os.path.join(self.workspace_path, output_file), 'r') as f:
                report_data = json.load(f)
            return {
                "success": True,
                "report": report_data
            }
        return {"success": False, "stderr": result.stderr}
