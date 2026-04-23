import os
import subprocess
import json

class EnvManagerTool:
    """Tool for managing Python Virtual Environments safely."""
    
    def __init__(self, venv_path):
        self.venv_path = venv_path
        self.pip_path = os.path.join(venv_path, "bin", "pip")
        self.python_path = os.path.join(venv_path, "bin", "python")

        if not os.path.exists(self.pip_path):
            raise FileNotFoundError(f"Virtual environment pip not found at {self.pip_path}")

    def list_packages(self):
        """Returns a list of installed packages and their versions."""
        result = subprocess.run([self.pip_path, "list", "--format=json"], capture_output=True, text=True)
        if result.returncode == 0:
            return json.loads(result.stdout)
        return {"error": result.stderr}

    def install_package(self, package_spec, use_cache=True):
        """
        Installs a specific package version (e.g., 'mindspore==2.3.0').
        Only uses cached wheels or trusted indices to prevent arbitrary downloads.
        """
        cmd = [self.pip_path, "install", package_spec]
        if use_cache:
            cmd.extend(["--prefer-binary"]) # Prefer cached/binary wheels

        result = subprocess.run(cmd, capture_output=True, text=True)
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    def uninstall_package(self, package_name):
        """Safely uninstalls a package."""
        cmd = [self.pip_path, "uninstall", "-y", package_name]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    def get_package_info(self, package_name):
        """Gets detailed info about an installed package."""
        cmd = [self.pip_path, "show", package_name]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        return f"Package '{package_name}' not found."
