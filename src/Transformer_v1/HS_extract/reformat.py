#!/usr/bin/env python3
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
import functools
import operator
import re
import pickle
import json

from tqdm import tqdm

print("Using configuration file : config.json")
config = json.loads(re.sub(r'#.*?\n', '', open('config.json', 'r').read()))

print('Read file : ', config['input_hts_data'])
hs = pd.read_csv(config["input_hts_data"], index_col=0)

def proc(s) :
    lst = s.split(".")
    s = lst[0] + lst[1]
    return s

hs['htsno1'] = hs.htsno.apply(lambda x: proc(x))

hs = hs[~hs.htsno1.apply(lambda x : True if re.search("^98\d{3}|^99\d{3}" , x) else False)]

tokenizer = RegexpTokenizer(r'\w+').tokenize
#tokenizer = word_tokenize # Don't use

def split_chunks(l, wb=32):
    d = len(l)
    for i in range(d):
        if (i % wb == 0) :  
            yield " ".join(l[i:min(i+wb, d)])

def clean_text(x) :
    y = ' '.join(list(x)).lower()
    y = re.sub(r"described.*provisions", "", y)
    y = y.replace('other', '')
    toks = tokenizer(y)
    y = list(split_chunks(toks))
    return y    

columns_to_merge = ["level_{}".format(s) for s in range(12)]

collect_dict_df = []
collect_dict = {}
train_dict = []

for g, d in tqdm(hs.groupby(['htsno1'])) :
    d = d.fillna('')
    feature_set = [set(list(d[x])) for x in columns_to_merge]
    feature_set = [clean_text(ele) for ele in feature_set if ele != {''}]
    feature_set = functools.reduce(operator.iconcat, feature_set, [])

    #print(feature_set)
    wc = list(map(lambda x : len(x.split()), feature_set))
    collect_dict_df.append({'hs': g , 'desc': feature_set, 'max_count' : max(wc), 'num_embeddings' : len(wc)})
    train_dict.append({'hs' : str(g), 'desc' : " ".join(feature_set)})
    collect_dict[g] = feature_set
    
print('Write file : ', config['hts_map_pkl'])
with open(config['hts_map_pkl'], 'wb') as f: pickle.dump(collect_dict, f)


train_dict_df = pd.DataFrame(train_dict)
print("Save labelled csv for training ", config['hts_train'])
train_dict_df.to_csv(config['hts_train'], index=False, header=True)



final_hts_df = pd.DataFrame(collect_dict_df)

print(final_hts_df.head())
print(">>>>> 10121")
print(list(final_hts_df[final_hts_df.hs == '10121'].desc))
print(">>>>> 611020")
print(list(final_hts_df[final_hts_df.hs == '611020'].desc))
print(">>>>> 611030")
print(list(final_hts_df[final_hts_df.hs == '611030'].desc))
