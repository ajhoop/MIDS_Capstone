# src 

The `src/` directory stores Python code used for deployment of models or running the final notebook in the `notebooks/final/` directory. 

# Transformer_v1 

This contains the version 1 or transformer model.  The tokenizer used BPE encoding. This worked well for small datasets, but performed poorly for large datasets. The final output layer implemented a clustering approach.

# Transformer_v2 (final version used in our MVP)

This is the final version of the custom transformer model. Contains the scripts used to train and test the final model.
