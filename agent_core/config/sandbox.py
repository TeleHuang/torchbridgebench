# Agent Sandbox and Constraints
# This file defines the explicit boundaries of what the Diagnostic Agent is allowed to do.

SANDBOX_CONFIG = {
    "ALLOWED_DIRECTORIES": [
        "/root/autodl-tmp/torchbridgebench",
        "/root/autodl-tmp/mindnlp/src/torch4ms",
        "/root/autodl-tmp/mindnlp/.venv-torch4ms",
        "/root/autodl-tmp/mindnlp/.venv-torch4ms-clean"
    ],
    
    "BLOCKED_DIRECTORIES": [
        "/",
        "/etc",
        "/var",
        "/root/.ssh",
        "/usr/local/Ascend" # Prevent modifying system NPU drivers
    ],
    
    "ALLOWED_COMMANDS": [
        "pip install",
        "pip uninstall",
        "pip list",
        "pip show",
        "python",
        "grep",
        "cat",
        "ls"
    ],
    
    "BLOCKED_COMMANDS": [
        "rm -rf",
        "sudo",
        "chmod",
        "chown",
        "apt-get",
        "yum",
        "reboot",
        "shutdown"
    ],
    
    "MUTATION_RULES": {
        "MAX_RETRIES": 3, # Maximum number of environment mutations per diagnostic session
        "ALLOW_DOWNGRADE": True,
        "REQUIRE_BACKUP": True # Must snapshot `pip freeze` before mutating
    }
}

def validate_path(path):
    """Ensure the path is within allowed directories and not in blocked directories."""
    path = os.path.abspath(path)
    
    for blocked in SANDBOX_CONFIG["BLOCKED_DIRECTORIES"]:
        if path.startswith(blocked) and not any(path.startswith(allowed) for allowed in SANDBOX_CONFIG["ALLOWED_DIRECTORIES"]):
            raise PermissionError(f"Agent is restricted from accessing {path}")
            
    if not any(path.startswith(allowed) for allowed in SANDBOX_CONFIG["ALLOWED_DIRECTORIES"]):
        raise PermissionError(f"Path {path} is outside the allowed sandbox.")
    
    return True
