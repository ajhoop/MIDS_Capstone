#!/usr/bin/env python3
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle

import json
import re


from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from sklearn.model_selection import train_test_split

print("Using configuration file : config.json")
config = json.loads(re.sub(r'#.*?\n', '', open('config.json', 'r').read()))

# Dataset setup
sampled_df  = pd.read_csv("csv/train.csv")

random_seed = 1
all_classes = sampled_df.HS_Code.unique()
print('DEBUG', all_classes)

sampled_df = sampled_df[['HS_Code', 'Product Desc']].rename({'HS_Code' : 'label', 'Product Desc' : 'text'}, axis=1)

sampled_df.text = sampled_df.text.apply(lambda x : x.lower())
sampled_df.text = sampled_df.text.apply(lambda x : x.replace('<br/>', ''))

all_train_df = []
all_valid_df = []
for c in all_classes :
    df = sampled_df[sampled_df.label == c]
    test_df = df.sample(frac=0.33333, random_state=random_seed)
    valid_df = df.drop(test_df.index)
    all_train_df.append(test_df)
    all_valid_df.append(valid_df)

train_csv_df = pd.concat(all_train_df)
test_csv_df  = pd.concat(all_valid_df)

train_csv_df = train_csv_df.reset_index().drop(['index'], axis=1)
train_csv_df['Index'] = train_csv_df.reset_index().index

test_csv_df = test_csv_df.reset_index().drop(['index'], axis=1)
test_csv_df['Index'] = test_csv_df.reset_index().index

## Labelled CSV file
print("Save labelled csv for training ", config['train_csv'])
train_csv_df.to_csv(config['train_csv'], index=False, header=True)  

# Labelled test CSV file
print("Save labelled csv for inference ", config['test_csv'])
test_csv_df.to_csv(config['test_csv'], index=False, header=True)  

#exit()

print('Reading the files : ' , config['training_objs'])
data_to_train = []
for f in config['training_objs'] :
  with open(f, 'rb') as f: training_data = pickle.load(f)
  data_to_train = data_to_train + training_data

# Text file to train data
print("Write text file for training ", config['training_file'])
with open(config['training_file'], 'w') as f:
    for item in data_to_train:
        f.write("{}\n".format(item))

tokenizer = Tokenizer(BPE())


tokenizer.pre_tokenizer = Whitespace()


# Read back text
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train(files=[config['training_file']], trainer=trainer)

print("Use padding length ", config['padding_length'])
tokenizer.enable_padding(length=int(config['padding_length']))

# Save tokenizer
print("Save tokenizer ", config['token_config'])
tokenizer.save(config['token_config'])
tokenizer = Tokenizer.from_file(config['token_config'])

# Save the training data as pickle
print("Save training data for training LM as pickle ", config['training_data_pkl'])
with open(config['training_data_pkl'], 'wb') as f: pickle.dump(data_to_train, f)
print("Length of LM training data ", len(data_to_train))
