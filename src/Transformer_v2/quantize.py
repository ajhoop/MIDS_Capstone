#!/usr/bin/env python3
import IPython

import sys
import os
import json
import re
sys.path.insert(0, '.')
import train
import torch
from tokenizers import Tokenizer
import pandas as pd
import numpy as np
import pickle


config = json.loads(re.sub(r'#.*?\n', '', open('config.json', 'r').read()))
model  = train.HTSClassifier()
checkpoint = torch.load(config['lm_save_file'])
model.load_state_dict(checkpoint['state_dict'])

tokenizer = Tokenizer.from_file(config['token_config'])
padding_length = int(config['padding_length'])
with open(config['save_dir'] + '/label_enc.pkl', 'rb') as f: label_enc = pickle.load(f)
index_to_name = {i : l for i, l in enumerate(label_enc.classes_)}
with open(config['save_dir'] + '/index_to_name.pkl', 'wb') as f: pickle.dump(index_to_name, f)
#print(index_to_name, file=open(config['save_dir'] + '/index_to_name.json', 'w'))

pd.set_option('display.max_colwidth', None)

def get_sample_prediction(text, m, num_samples=10):
    enc = tokenizer.encode(text)
    ids = np.array(enc.ids[:padding_length])
    ids = np.vectorize(lambda x : 1 if not x else x)(ids)
    mask  = (torch.from_numpy(np.array(ids)) == 0)
    ids = torch.from_numpy(ids)
    y = m.forward(ids.reshape(1, padding_length), mask.reshape(1, padding_length))
    logits = torch.softmax(y, dim=1)
    sorted_prob, indices = torch.sort(logits, descending=True)

    sorted_prob, indices = sorted_prob.detach(), indices.detach()
    indices = np.vectorize(index_to_name.get)(indices[0].numpy()[:num_samples])

    sorted_prob = sorted_prob[0].numpy()[:num_samples]
    df_rank = pd.DataFrame([{'hs' : c, 'probablity' : p} for c, p in zip(indices, sorted_prob)])
    return df_rank

# ----- Quantize the model

quantized_model = torch.quantization.quantize_dynamic(
    model.eval(), {torch.nn.Linear}, dtype=torch.qint8
)
print(quantized_model)

def print_size_of_model(model, s):
    torch.save(model.state_dict(), "temp.p")
    print(s + ' Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

print_size_of_model(model, "Original model")
print_size_of_model(quantized_model, "Quantized")

# ----- Save the model

ids = np.random.randint(5, 100, padding_length)
mask  = (torch.from_numpy(np.array(ids)) == 0)
ids = torch.from_numpy(ids)
dummy_input = (ids.reshape(1, padding_length), mask.reshape(1, padding_length))

traced_model = torch.jit.trace(quantized_model, dummy_input)
torch.jit.save(traced_model, config['save_dir'] + "/transformer_quant.pt")

#IPython.embed(); exit(1)

# ---- Test
# https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html

query = "orange juice"
print("+++++ test with query : ", query)

df = get_sample_prediction(str(query), model, 10)
#print("\nOriginal model\n", df)
loaded_quantized_model = torch.jit.load(config['save_dir'] + "/transformer_quant.pt")
df = get_sample_prediction(str(query), loaded_quantized_model, 10)
#print("\nQuantized model\n", df)

def checkEqual(L1, L2): return (len(L1) == len(L2) and sorted(L1) == sorted(L2))
def myfunc(msg='assert OK'): 
    print(msg)
    return True
assert checkEqual(list(df.hs), list(df.hs)) and myfunc("Quantization test passed"), "Quantization test failed"
