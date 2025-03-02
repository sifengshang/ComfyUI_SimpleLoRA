# ComfyUI_SimpleLoRA

**<u>'ComfyUI_SimpleLoRA'</u>** is a custom node of ComfyUI that enables you to LoRA fine-tune stable diffusion models. It is simple because you can easily apply LoRA fine-tuning to the base model through **configuring only a single node 'LoRATrainer'**. It also **supports training with either a HuggingFace dataset or a custom dataset constructed by yourself**. After training, the LoRA weights can be smoothly identified and loaded with the built nodes such as 'LoadLoRA' or 'LoRALoaderModelOnly'.

To be more specific, the node starts a text-to-image LoRA fine-tuning script (`lora_core.py`) modified from the [example of diffusers](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py), and you can configure the training arguments through the 'LoRATrainer' node. LoRAs are applied to linear modules in cross attention layers of UNet. The ranks are by default set to 4, and could be easily modified through the trainer node. We marked in our codes where was modified in the annotations. Search the keyword 'SimpleLoRA Added' in `lora_core.py` for more details if you are interested in coding.

This custom node is developed as a LoRA showcase of the course **COMP3076: AI and Generative Arts, HKBU**. The developer is one of the teaching assistants of this course during semeter 2024-2025 S2. The node will be continually refined and updated to suit the needs of the course. You are also welcome to download it now and give it a try. If you encouter some issues or bugs during using the node or have some suggestions for improvements, feel free to open an issue in GitHub and let me know ;)

If you like this custom node, please give me a star🤗

## Installation

To install SimpleLoRA, one may first download the zip file of the source codes and exact it under the folder `ComfyUI\custom_nodes`. You can also open a terminal/CMD window inside the folder `ComfyUI\custom_nodes`, and install the node using git:

```shell
git clone https://github.com/sifengshang/ComfyUI_SimpleLoRA.git
```

To install the Python dependencies of the SimpleLoRA, you can enter the root folder of the node `ComfyUI\custom_nodes\ComfyUI_SimpleLoRA`, and double click to execute the batch file `install_requirements.bat`. It will call the `pip` tool of the embeded Python in ComfyUI (`python_embeded\python.exe`) and automatically install all the required dependencies. Alternatively, you can manully install dependencies specified in `requirements.txt` with the embeded ComfyUI `pip`.

## How to use

### Build the workflow

Open your ComfyUI and create an **empty workflow**. Add the node 'LoRATrainer' by right-clicking your mouse and sequentially select 'Add Node', 'SimpleLoRA' and 'LoRATrainer', or directly search 'LoRATrainer' using the search tool. The workflow containing **only a single node** should look like this:

![](.\figures\workflow.jpg)

### Configure the node

#### Base model

The very first parameter 'ckpt_name' reads the base models inside the folder `ComfyUI\models\checkpoints`. You can select a basic **text-to-image** base model to apply the LoRA, such as [DreamShaper 8](https://huggingface.co/Lykon/DreamShaper/tree/main). Currently I only tried to fine-tune base models with SD1.5 architecture, but it should also support other stable diffusion families with UNet comopnent as well.

#### Dataset

To specify a dataset for LoRA fine-tuning, you need to configure three parameters jointly: 'use_custom_dataset', 'dataset_path' and 'caption_column'.

For **HuggingFace datasets**, please set 'use_custom_dataset' to 'false', and copy the dataset name (e.g., [svjack/pokemon-blip-captions-en-zh](https://huggingface.co/datasets/svjack/pokemon-blip-captions-en-zh)) to 'dataset_path'. The node only supports datasets for **text-to-image** task. The specific column name of image captions may vary among different datasets. You may need to look it up in the Dataset Viewer on the HuggingFace website of the dataset, and put the selected column name to the parameter 'caption_column'. For example, if you want to use English captions of the dataset 'pokemon-blip-captions-en-zh', you should put 'en_text' for 'caption_column'.

For **custom datasets**, you should of course construct one first. The root directory of the custom dataset should look like this:

```
.
├── image
│   ├── 0.png
│   ├── 1.png
│   └── ...
└── text
    ├── 0.txt
    ├── 1.txt
    └── ...
```

It should be noted that any [image format supported by HuggingFace datasets](https://github.com/huggingface/datasets/blob/main/src/datasets/packaged_modules/imagefolder/imagefolder.py#L40) can be accepted, and it is not necessary, though recommended, to name the image filename with numbers starting from 0. The node itself will automatically pair the image and text file (captions) having the same name, construct a [ImageFolder](https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder) class to process the dataset, and generate an annotation file 'metadata.jsonl' inside the `image` folder.

After constructing the custom dataset, you may set 'use_custom_dataset' to 'true', paste the root directory of the custom dataset into 'dataset_path', and leave the parameter 'caption_column' as its default value 'text'.

#### Checkpointing

During training, checkpoints of the LoRA weights will be automatically saved into the folder `ComfyUI\models\lora_logs\[lora_checkpoint_folder]`. When the LoRA fine-tuning is finished, the final LoRA weights will be automatically placed into the folder `ComfyUI\models\loras` with a name `[lora_name].safetensors`.

It should be noted that despite LoRA weights are checkpointed during training, the current version of SimpleLoRA does NOT support resume training a LoRA from checkpoints. This feature may be updated in future version.

#### Training Arguments

The rest of the parameters are quite straightforward. You can specify the initial learning rate (decayed by cosine annealing), batch size, rank of LoRAs, and the random seed directly through the parameters. The total steps of training is determined by the paramter 'training_step', and the LoRA weights will be checkpointed every 'checkpointing_steps'.

After setting are the parameters properly, you can start the training workflow by clicking the 'Queue' botton. 

### Apply the trained LoRA

If training successful, you can immediately apply the trained LoRA to the corresponding base model with the node 'LoadLoRA' or 'LoRALoaderModelOnly'. Since only the UNet is applied by LoRAs, it is recommended that you use the latter node. If you are not familiar with LoRA workflows, this [ComfyUI example](https://comfyanonymous.github.io/ComfyUI_examples/lora/) is suitable for your reference.

Below are some showcases of the trained LoRA.

![](.\figures\showcases.jpg)







