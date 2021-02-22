# README

## Dependency
- runs on GPU, CPU can be slow
- Huggingface tokenizer https://github.com/huggingface/tokenizers
- pytorch, torchvision

## Config 
This is being read in by all the scripts. All main hyperparameters are abstracted into here
./config.json

All common functions used by different stages are packaged here
./TransformerModel.py

## Steps to get inferences 

### (Step1) Prepare the dataset
- Input CSV is parsed. 
- Tokenizers are trained from scratch.  
- Training data and tokenizer are stored 

% ./prepare.py

### (Step2) Train the language model
- Do random mask training.
- Save the weights of the transformer component after training

% ./language_train.py

### (Step3) Fine tune the language model
- Load the weights from (step2)
- Fine tune is done on training split
- Objective is to create max separability between classes
- Save the weights

% ./fine_tune.py

### (Step4) Generate the embedding 
- Load the weights from (step3)
- Iterates through each example in training set and generate an embedding
./generate_embeddings.py

### (Step5) Final inference
- Load the weights from (step3)
- Load the embeddings from (step4)
- Generate per-class embedding
- Do a cosine compare of each test example against per-class embedding
- Do softmax + argmax to get ranking with probablity

% ./do_inference.py
