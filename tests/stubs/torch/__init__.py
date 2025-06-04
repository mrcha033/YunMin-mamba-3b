from types import ModuleType
import sys

class cuda:
    @staticmethod
    def is_available():
        return False
    @staticmethod
    def device_count():
        return 0
    @staticmethod
    def get_device_name(_):
        return "mock"

# Create utils.data submodule with DataLoader
utils = ModuleType("torch.utils")
data = ModuleType("torch.utils.data")
class DataLoader:
    pass

data.DataLoader = DataLoader
utils.data = data
sys.modules[__name__ + ".utils"] = utils
sys.modules[__name__ + ".utils.data"] = data
