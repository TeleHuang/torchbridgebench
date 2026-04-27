from .base import BaseAdapter

class Torch4MSAdapter(BaseAdapter):
    @property
    def name(self):
        return "torch4ms"

    def setup(self):
        import os
        import sys
        repo_root = os.environ.get(
            "TORCH4MS_REPO_ROOT",
            "/root/autodl-tmp/ascend-torch4ms-ms272-stable",
        )
        if os.path.isdir(repo_root) and repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        # If torch4ms was imported from another location, reload from the repo root.
        old_mod = sys.modules.get("torch4ms")
        old_file = getattr(old_mod, "__file__", "") if old_mod is not None else ""
        if old_mod is not None and old_file and not old_file.startswith(repo_root):
            for mod_name in list(sys.modules.keys()):
                if mod_name == "torch4ms" or mod_name.startswith("torch4ms."):
                    sys.modules.pop(mod_name, None)
        import torch4ms
        # Force torch4ms runtime to Ascend so benchmark reflects NPU behavior.
        # Environment() defaults to CPU when no explicit target is provided.
        cfg = torch4ms.config.Configuration()
        cfg.default_device_target = os.environ.get("TORCH4MS_DEVICE_TARGET", "Ascend")
        cfg.use_ms_graph_mode = bool(int(os.environ.get("TORCH4MS_USE_MS_GRAPH_MODE", "0")))
        torch4ms.env = torch4ms.tensor.Environment(configuration=cfg)
        self.env = torch4ms.env
        self.env.__enter__()

    def run_module(self, module, *args, **kwargs):
        # 保持原生 PyTorch 语义：是否追踪梯度由用户代码决定。
        return module(*args, **kwargs)

    def teardown(self):
        if hasattr(self, 'env') and self.env:
            self.env.__exit__(None, None, None)
