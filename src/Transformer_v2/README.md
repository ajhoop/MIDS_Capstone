# Custom Transformer training

## Below is a description of various files in this directory 

- `TransformerModel.py` - Basic transformer model used in the training. Uses pytorch `nn.TransformerEncoderLayer` and `nn.TransformerEncoder`. Please refer to pytorch documentation for more details. This is just a wrapper to package various components.

- `config.json`  - Captures various configurable parameters in the model training. The parameters are being used by various steps. Keeping them separately reduces  mismatch between various stages. 

- `create_archive` - Used for deployment testing along with quantization. Shell script for creating the mar archive file used by `torchserv`. Refer to `torchserv` documentation for further details.

- `do_inference_single.py` - Used for single sample inference testing for the model.

- `launch_inf` : Shell script file for local deployment of model with Docker. Contains the curl commands used for testing.

- `model_handler.py` - defines a custom model handler as required by torchserv.

- `prepare.py` -  Prepare train test split for training the Custom transformer model. This does feature extraction and feature engineering.  The output from this script is saved into the data folder. The files names saved are controlled through `config.json`. 
 
- `quantize.py` - This performs the model quantization.  The size of the model is reduced to ~19Mb after quantization, without any loss in prediction accuracy.

- `train.py` - Main training script

- `deploy_transformer_v2.ipynb` - This is used as a simple web interface to test deployment of  transformer models. Need holoviews and Bokeh to run this. 

- `hts_train.csv` - Used by the `deploy_transformer_v2.ipynb`, to list the description of relevant HS code.

- `Commodity_hts_extract.csv` - This is the auxiliary dataset provided by Government dataset. This is used as an add on to training.


## Tools needed.

Standard datascience tools - `pandas`, `numpy`, `nltk` and associated `nltk` packages.  For visualization `holoviews` need to be installed.  The visualization can be run in Jupyter notebook.

## Neural Network Training tools 

`pytorch` and `pytorchLightning`. Refer to `pytorch` and `pytorchLightning` documentation for more details

## Hardware

A machine with nVidia GPU (Used RTX2070 here) is needed.  This can be run in AWS or in a local machine with GPU.

## Software

Ubuntu 18.04+ is needed. Stick to LTS versions

## Training time

Around 2 hours. 

## Matrics

All the log information is piped into wandb.ai. The specifics are configured through `config.json`. The graphs are updated live as training proceeds. Refer to wandb.ai documentation for more details

## Steps

```bash
# This will populate the data directory with all required files for training
% ./prepare.py

# This will train the network. The weights will be stored in the data directory
% ./train.py

# This will trigger the quantization process.
% ./quantize.py

# Will create the archieve for deployment testing
% ./create_archive

# This will deploy the model
% ./launch_inf

```

Open the note book `deploy_transformer_v2.ipynb` and execute the same from a Jupyter/JupterLab environment. This will enable the user to input the prediction and see the predicted results. 


## output from `wandb.ai`


Below are the outputs of the runs in wandb.ai.


| Run                                | Link                                             |
|------------------------------------|--------------------------------------------------|
| 4137 classes, local GPU            | https://wandb.ai/shajikk/full_1000/runs/11pg25v5 |
| 4137 classes, AWS                  | https://wandb.ai/shajikk/full_1000/runs/3onw0hrx |
| 5386 classes, local machine        | https://wandb.ai/shajikk/full_1000/runs/2he7od4j |
| 209 classes (medium) local machine | https://wandb.ai/shajikk/full_1000/runs/336zbq6c |
| 10 classes (small) local machine   | https://wandb.ai/shajikk/full_1000/runs/3kb6ys8x |




