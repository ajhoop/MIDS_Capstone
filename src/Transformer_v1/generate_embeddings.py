#!/usr/bin/env python3

import IPython
import sys  
import pandas as pd
import transformers

sys.path.insert(0, '.')
import TransformerModel
import pickle
import json
import re

print("Using configuration file : config.json")
config = json.loads(re.sub(r'#.*?\n', '', open('config.json', 'r').read()))

retrofit_trained = TransformerModel.Misc()
df_train = pd.read_csv(config['train_csv'], dtype={'label': str, 'Index' : int})
retrofit_trained.get_embedding(df_train)

