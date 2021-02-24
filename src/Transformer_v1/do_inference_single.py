#!/usr/bin/env python3
import IPython
import sys  
import pandas as pd
sys.path.insert(0, '.')
import TransformerModel
import json
import re
import numpy as np
import pickle

np.set_printoptions(precision=30)


class DoInference():

    def __init__(self) :
       self.config = json.loads(re.sub(r'#.*?\n', '', open('config.json', 'r').read()))
       self.predict = TransformerModel.Predict()
       self.hts_map = pd.read_csv(self.config['hts_map'], dtype={'hs': str, 'desc' : str})

    def do_inference(self, text, number=5):
       self.text = text
       df_val   = pd.DataFrame([{'label' : 'None', 'text' : self.text, 'Index' : 0 }])
       self.predict.setup_dataloader(df_val)
       prob = self.predict.do_inference(method = 'prob')
       self.df_rank = pd.DataFrame({'hs' : list(prob[0][0]), 'probablity' : list(prob[1][0]), 'cosine' : list(prob[2][0]) })
       self.df_rank  = self.df_rank.merge(self.hts_map, on='hs', how='left').fillna('No description')
       return self.df_rank.head(number)

#i = DoInference()
#i.do_inference("cycle")
