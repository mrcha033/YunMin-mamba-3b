class Accelerator:
    def __init__(self, *args, **kwargs):
        self.device = 'cpu'
    def prepare(self, obj):
        return obj
    def save_state(self, path):
        pass
