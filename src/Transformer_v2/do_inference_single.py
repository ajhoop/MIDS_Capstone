#!/usr/bin/env python3
import IPython
import sys  
import pandas as pd
sys.path.insert(0, '.')
import TransformerModel
import train
import json
import re
import numpy as np
import pickle
from tokenizers import Tokenizer
import torch

#np.set_printoptions(precision=30)


class DoInference():

    def __init__(self) :

       self.config = json.loads(re.sub(r'#.*?\n', '', open('config.json', 'r').read()))
       #self.model = HTSClassifier().eval().cuda(device=0)
       self.model  = train.HTSClassifier.load_from_checkpoint(self.config['lm_save_file']).eval()
       self.tokenizer = Tokenizer.from_file(self.config['token_config'])
       self.padding_length = int(self.config['padding_length'])
       self.hts_map = pd.read_csv("hts_train.csv", dtype={'hs': str, 'desc' : str})
       with open(self.config['save_dir'] + '/label_enc.pkl', 'rb') as f: self.label_enc = pickle.load(f)

    def do_inference(self, text, number=10):
        with torch.no_grad():
             enc = self.tokenizer.encode(text)
             ids = np.array(enc.ids[:self.padding_length])
             ids = np.vectorize(lambda x : 1 if not x else x)(ids)
             mask  = (torch.from_numpy(np.array(ids)) == 0)
             ids = torch.from_numpy(ids)
             
             y = self.model.forward(ids.reshape(1, self.padding_length), mask.reshape(1, self.padding_length))
             logits = torch.softmax(y, dim=1)
             sorted_prob, indices = torch.sort(logits, descending=True)

             indices = self.label_enc.inverse_transform(indices[0].numpy()[:number])
             sorted_prob = sorted_prob[0].numpy()[:number]
             
             df_rank = (pd.DataFrame([{'hs' : c, 'probablity' : p} for c, p in zip(indices, sorted_prob)])
                          .merge(self.hts_map, on='hs', how='left').fillna('No description')
                       )
             IPython.embed(); exit(1)
        return 1

i = DoInference()
i.do_inference("cycle chains")
