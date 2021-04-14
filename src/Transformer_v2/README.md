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




