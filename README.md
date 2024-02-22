# MeLLM
The code and data of paper "MeLLM: Image-to-Text Generation via Frozen Multimodal Encoders and Large Language Models"

## Setting Environment
To successfully setup the environment, you are required to install the package mentioned in 'requirement.txt'. 

We also recommand you to create a virtual envieronment and install the required packages by running the following script: 
```bash
conda create -n MeLLM python=3.9
conda activate MeLLM
pip install -r requirement.txt
```
## Checkpoint of Mutimodal Encoders
For OFA and BEiT3 encoder checkpoints, they can be obtained from the below wonderful projects:

OFA: https://github.com/OFA-Sys/OFA

BEiT3: https://github.com/microsoft/unilm/tree/master/beit3

## Checkpoint of LLM
The checkpoint of LLaMA2 and Vicuna can be obtained from their corresponding huggingface projects:

LLaMA2:https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

Vicuna v1.5:https://huggingface.co/lmsys/vicuna-7b-v1.5

## Lauching Training
Specify the checkpoints of multimodal encoders and LLM at 'run.sh', after that, you are able to launch the training by running it
```bash
sh run.sh
```

