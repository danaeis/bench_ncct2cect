# Welcome to CFPS-Diff!
**Contrast Flow Pattern and Cross-Phase Specificity-Aware Diffusion Model (CFPS-Diff)** is a generative diffusion model that can synthesize multiphase CECT  from NCCT.


This repository contains the code to our paper "Contrast Flow Pattern and Cross-Phase Specificity-Aware Diffusion Model for NCCT-to-Multiphase CECT Synthesis".

# Results
<img src="https://github.com/Kindyz/CFPS-Diff/blob/main/Visualization.png" width="800px">

# Setup Environment
In order to run our model, we suggest you create a virtual environment
```
conda create -n CFPS-Diff_env python=3.8
```
and activate it with
```
conda activate CFPS-Diff_env
```
Subsequently, download and install the required libraries by running
```
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt




# Training
First, you need to train the condictional diffusion model. To do so in prepared dataset, you can run the following command:
```
python ./main/train_CFPS-Diff_gen.py
You can use the following command to observe the loss curve of the training process, visualize the sample image, etc.
```
tensorboard --lodgir ./main/trained_models/CFPS-Diff_gen/(model save filename)
```
[Supplement] Problem troubleshooting can be found in Error_troubleshooting.txt
# Inference
In the inference stage, synthesis and identification are performed together. You can do this by running the following command:
```
python ./main/Inference.py
```


