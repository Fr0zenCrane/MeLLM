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

## Lauching Training
Specify the checkpoints of multimodal encoders and LLM at 'run.sh', after that, you are able to launch the training by running it
```bash
sh run.sh
```

