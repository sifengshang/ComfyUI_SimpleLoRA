import os
import os.path as osp
import errno
import subprocess

import folder_paths

from safetensors import safe_open
from safetensors.torch import save_file

class LoRATrainer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {    
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "The name of the checkpoint (model) to load."}), # base model, dataset_dir, training configs(hyperparameters)
                "use_custom_dataset": ("BOOLEAN", {
                    "defualt": False
                }),
                "dataset_path": ("STRING", {
                    "multiline": False, 
                    "default": "" # svjack/pokemon-blip-captions-en-zh
                }),
                "lora_checkpoint_folder": ("STRING", {
                    "multiline": False, 
                    "default": "lora_test"
                }),
                "lora_name": ("STRING", {
                    "multiline": False, 
                    "default": "my_lora"
                }),
                "caption_column": ("STRING", {
                    "multiline": False, 
                    "default": "text"
                }),
                "learning_rate": ("FLOAT", {"default": 1e-4, "min": 0.0, "step": 1e-5}),
                "batch_size": ("INT", {"default": 1, "min":1}),
                "training_steps": ("INT", {"default": 500, "min":1}),
                "checkpointing_steps": ("INT", {"default": 100, "min":1}),
                "rank": ("INT", {"default": 4, "min": 1,}),
                "random_seed": ("INT", {"default": 0})
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()

    FUNCTION = "lora"
    OUTPUT_NODE = True
    CATEGORY = "SimpleLoRA"

    def lora(self, ckpt_name, 
             use_custom_dataset, dataset_path, 
             lora_checkpoint_folder, lora_name,
             caption_column,
             learning_rate, batch_size, training_steps, checkpointing_steps, rank, random_seed):
        def mkdir_if_missing(dirname):
            """Create dirname if it is missing."""
            if not osp.exists(dirname):
                try:
                    os.makedirs(dirname)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise

        models_dir = folder_paths.models_dir
        model_path = osp.join(models_dir, 'checkpoints', ckpt_name)

        base_path = folder_paths.base_path
        base_dir = osp.dirname(base_path)
        python_path = osp.join(base_dir, 'python_embeded', 'python.exe')

        print("#"*50)
        print("#")
        print("# Using the embedded python interpreter to start the LoRA training script:")
        print(f"# {python_path}")
        print("#")
        print("#"*50)

        this_path = osp.abspath(__file__)
        this_dir = osp.dirname(this_path)
        lora_path = osp.join(this_dir, 'lora_core.py')
        
        output_dir = osp.join(models_dir, "lora_logs", lora_checkpoint_folder)
        mkdir_if_missing(output_dir)
        # Turn Boolean to String to avoid bugs in argparse
        use_custom_dataset = "False" if not use_custom_dataset else "True"

        command = f'{python_path} -m accelerate.commands.launch ' + '--mixed_precision="bf16\" --num_processes=1 ' + f'{lora_path} ' \
                  + f'--pretrained_model_name_or_path="{model_path}" ' \
                  + f'--dataset_name="{dataset_path}" ' \
                  + f'--use_custom_dataset={use_custom_dataset} ' \
                  + f'--train_batch_size={batch_size} ' \
                  + f'--output_dir="{output_dir}" ' \
                  + f'--rank={rank} ' \
                  + '--dataloader_num_workers=0 ' + '--resolution=512 ' + '--center_crop ' + '--random_flip ' \
                  + '--gradient_accumulation_steps=4 ' \
                  + f'--max_train_steps={training_steps} ' \
                  + f'--learning_rate={learning_rate} ' \
                  + '--max_grad_norm=1 ' \
                  + '--lr_scheduler="cosine" ' + '--lr_warmup_steps=0 ' \
                  + f'--checkpointing_steps={checkpointing_steps} ' + f'--caption_column="{caption_column}" ' \
                  + f'--seed={random_seed} ' + '--use_8bit_adam'
                  
        subprocess.run(command, shell=True)


        print("#"*50)
        print("#")
        print("# Training is finished!")
        print("#")
        print("#"*50)


        # Final Step:
        # Change the keys of the LoRA's .safetensors file,
        # making them recognizable by ComfyUI's LoadLoRAModelOnly node
        org_lora_path = osp.join(output_dir, 'pytorch_lora_weights.safetensors')
        lora_folder = osp.join(models_dir, 'loras')
        if not lora_name.endswith('.safetensors'):
            lora_name += '.safetensors'
        
        new_safetensors = {}
        with safe_open(org_lora_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                # print(key)
                # key_splits = key.split('.')
                # prefix = key_splits[:-3]
                # suffix = key_splits[-3:]
                # new_key = ['lora']
                # new_key.extend(prefix)
                # new_key = '_'.join(new_key)
                # new_key = new_key + '.' + suffix[0] + '_' + suffix[1] + '.' + suffix[2]

                new_key = 'lora_' + key.replace('.', '_').replace('A', 'down').replace('B', 'up')
                new_key_splits = new_key.split('_')
                new_key = '_'.join(new_key_splits[:-3]) + '.' + new_key_splits[-3] + '_' + new_key_splits[-2] + '.' + new_key_splits[-1]
                new_safetensors[new_key] = f.get_tensor(key)

        save_file(new_safetensors, osp.join(lora_folder, lora_name))

        print("#"*50)
        print("#")
        print("# Keys of LoRA's .safetensors file have been changed")
        print("# The LoRA can be found and loaded by the LoadLoRAModelONly Node")
        print("# You can find the trained LoRA's .safetensors file at: ")
        print(f"# {osp.join(lora_folder, lora_name)}")
        print("#")
        print("#"*50)

        return [False] # dummy output to avoid exceptions


class ChangeLORAKeys():
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {    
                "org_lora_path": ("STRING", {
                    "multiline": False, 
                    "default": ""
                }),
                "lora_new_name": ("STRING", {
                    "multiline": False, 
                    "default": "my_lora"
                }),
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()

    FUNCTION = "change_keys"
    OUTPUT_NODE = True
    CATEGORY = "SimpleLoRA/utils"

    def change_keys(self, org_lora_path, lora_new_name):
        models_dir = folder_paths.models_dir
        if not lora_new_name.endswith('.safetensors'):
            lora_new_name += '.safetensors'
        lora_folder = osp.join(models_dir, 'loras')
        new_safetensors = {}
        with safe_open(org_lora_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                # print(key)
                # key_splits = key.split('.')
                # prefix = key_splits[:-3]
                # suffix = key_splits[-3:]
                # new_key = ['lora']
                # new_key.extend(prefix)
                # new_key = '_'.join(new_key)
                # new_key = new_key + '.' + suffix[0] + '_' + suffix[1] + '.' + suffix[2]
                new_key = 'lora_' + key.replace('.', '_').replace('A', 'down').replace('B', 'up')
                new_key_splits = new_key.split('_')
                new_key = '_'.join(new_key_splits[:-3]) + '.' + new_key_splits[-3] + '_' + new_key_splits[-2] + '.' + new_key_splits[-1]
                new_safetensors[new_key] = f.get_tensor(key)

        save_file(new_safetensors, osp.join(lora_folder, lora_new_name))
        return [False]