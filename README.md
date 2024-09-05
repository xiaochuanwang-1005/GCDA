# GCDA

This is the implementation for  our paper "Self-Supervised Graph Contrastive Learning with Diffusion Augmentation for Functional MRI Analysis and Brain Disorder Detection (GCDA)". Code developed and tested in Python 3.9.17 using PyTorch 2.1.2+cu118. 

## Package Structure 

File details in GCDA_code.zip:

    - `./pretraining.py`: The pre-training of the proposed GCDA.
    - `./pretext_model.py`: The pretext model for pre-training mainly include graph diffusion augmentation (GDA) and graph contrastive learning.
    - `./diffusion_model.py`: The main functions for the GDA module mainly include noise unit and denoising neural network.
    - `./noisy_schedule.py`: The transition function in noise unit.
    - `./transformer_model.py`: The denoising neural network.
    - `./GIN_encoder.py`: The graph feature extraction backbone. 
    - `./diffusion_utils.py`: These are some useful functions in the GDA module. 
    - `./diffusion_loss`: This is diffusion loss function.
    - `./extra_features`: The calculating functions for global feature. 
    - `./extra_features1`: The copy of the extra_features for computing input dimensions of the denoising neural network. 
    - `./dataset`: Data preparation for pre-training. 
    - `./fine_tune`: The fine-tuning of the proposed GCDA.
    - `./fine_tune_model.py`: The task-specific model for fine-tuning.
    - `./dataset1`: Data preparation for fine-tuning. 

## Requirements

To perform GCDA training, some major requirements are given below:

```python
numpy=1.24.3\
scipy=1.10.1\
torch=2.1.2+cu118\
einops=0.7.0\
torchmetrics=1.1.1\
wandb=0.15.10
```

More details about arguments are concluded in the paper and code.


## Dataset
    
  We used the following datasets:
 
- ABIDE (Can be downloaded [here](http://fcon_1000.projects.nitrc.org/indi/abide/))
- REST-meta-MDD (Can be downloaded [here](http://rfmri.org/REST-meta-MDD))

## Contact

If you have any questions about the code of GCDA, please contact me through ``xiaochuan10052022@163.com``.




