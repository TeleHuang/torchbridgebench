from abc import ABC, abstractmethod

class BaseAdapter(ABC):
    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def setup(self):
        """Setup the backend environment, apply patches if necessary."""
        pass

    @abstractmethod
    def teardown(self):
        """Clean up the backend environment."""
        pass
    
    def run_operator(self, op_func, *args, **kwargs):
        """Run a single operator and return the result."""
        return op_func(*args, **kwargs)

    def run_module(self, module, *args, **kwargs):
        """Run a neural network module."""
        return module(*args, **kwargs)

    def run_model(self, model, *args, **kwargs):
        """Run a complete model."""
        return model(*args, **kwargs)
