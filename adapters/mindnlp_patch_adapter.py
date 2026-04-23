from .base import BaseAdapter

class MindNLPPatchAdapter(BaseAdapter):
    @property
    def name(self):
        return "mindnlp_patch"

    def setup(self):
        import mindnlp
        from mindnlp.patch import apply_all_patches
        apply_all_patches()

    def teardown(self):
        pass
