'''
+---------------------------------+
| Simple LoRA training in ComfyUI |
|                                 |
+---------------------------------+
'''

from .SimpleLoRA import LoRATrainer, ChangeLORAKeys

NODE_CLASS_MAPPINGS = {"LoRATrainer": LoRATrainer, "ChangeLoRAKeys": ChangeLORAKeys}

__all__ = ['NODE_CLASS_MAPPINGS']