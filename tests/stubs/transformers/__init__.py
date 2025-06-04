class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

class AutoConfig:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

class AutoModelForCausalLM:
    @classmethod
    def from_config(cls, *args, **kwargs):
        return cls()

class DataCollatorForLanguageModeling:
    pass
