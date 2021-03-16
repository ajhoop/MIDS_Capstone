#!/usr/bin/env python3
import IPython
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import numpy as np
import pickle

import json
import re
import random

import dask.dataframe as dd
from dask.diagnostics import ProgressBar
ProgressBar().register()

import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from nltk.corpus import wordnet as wn


from nltk.corpus import stopwords
stop = stopwords.words('english')
import re
from  inflect import engine
import itertools

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder 

import tokenizers
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from collections import ChainMap


print("Using configuration file : config.json")
config = json.loads(re.sub(r'#.*?\n', '', open('config.json', 'r').read()))

save_dir = config['save_dir']

# ++++++++++++++   Initial steps
# Read in the Cleaned parquet and convert it into panda frame
result_dd = dd.read_parquet('data/full_cleaned')
all_labels = result_dd.label.unique()
print('Number of classes : ', len(all_labels))

result_df = result_dd.compute()

print('Length of data frame', result_df.shape)


# ++++++++++++++  Cleanup routines
lemmatizer = WordNetLemmatizer()

# function to convert nltk tag to wordnet tag
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

from nltk.tokenize import RegexpTokenizer
tok = RegexpTokenizer(r'\w+')

def transform_word(word) :
    temp_list = []
    temp_list.append(engine().singular_noun(word))
    temp_list.append(engine().plural_adj(word))
    temp_list.append(engine().plural(word))
    temp_list.append(engine().plural_verb(word))
    return [w for w in list(set(temp_list)) if w]

    
def clean_sentence(sentence):
    #tokenize the sentence and find the POS tag for each token
    sentence = re.sub(r'\d+', '', sentence)
    remove_dig_pun = tok.tokenize(sentence.lower())

    nltk_tagged = nltk.pos_tag(remove_dig_pun)  
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
            
    lemmatized_sentence = list(map((lambda x : x if x not in stop else ""), lemmatized_sentence))
    
    txfm_lemmatized_sentence = list(itertools.chain(*[transform_word(word) for word in lemmatized_sentence]))
    
    combined = lemmatized_sentence + txfm_lemmatized_sentence
    combined = list(set(combined))
    combined.sort()
    
    return " ".join(remove_dig_pun + combined)

# ++++++++ Read the HTS code mapping, clean up

print("HTS dataframe cleanup..")
recompute = False

if recompute :
    hts = pd.read_csv("./hts_train.csv", dtype={'hs': str, 'desc' : str})
    hts['clean'] = hts.desc.apply(lambda x : clean_sentence(x))
    hts = hts.rename({'hs' : 'label', 'clean' : 'text'}, axis=1)[['label', 'text']]
    hts = hts[hts['label'].notna()]
    hts.to_pickle(save_dir + "/hts.pkl")
else : 
    hts = pd.read_pickle(save_dir + "/hts.pkl")

print(list(hts.iloc[0]))

# ++++++++ Concat

result = pd.concat([result_df, hts])
all_labels = result.label.unique()
print('Number of classes after concat: ', len(all_labels))

# ++++++++ Read the NAICS code mapping, clean up
print("NAICS dataframe cleanup..")
recompute = False

if recompute :
    nacis = pd.read_csv("commodity_hts_extract.csv", dtype={'hts6': str, 'description_long' : str})
    nacis['clean'] = nacis.description_long.apply(lambda x : clean_sentence(x))
    nacis = nacis.rename({'hts6' : 'label', 'clean' : 'text'}, axis=1)[['label', 'text']]
    nacis = nacis[nacis['label'].notna()]
    nacis.to_pickle(save_dir + "/nacis.pkl")
else : 
    nacis = pd.read_pickle(save_dir + "/nacis.pkl")
    
print(list(nacis.iloc[0]))

# ++++++++ Prepare word net
all_words = {}

def transform_word_from_wordnet(word) :
    temp_list = [word]
    temp_list.append(engine().singular_noun(word))
    temp_list.append(engine().plural_adj(word))
    temp_list.append(engine().plural(word))
    temp_list.append(engine().plural_verb(word))
    final = [w for w in list(set(temp_list)) if w]
    return [a for a in final if not re.findall('[^A-Za-z]+', a)]

recompute = False
final_wd_dict = {}

if recompute :
    words = [word for word in wn.words()]
    for word in tqdm(words):
    
        all_words[word] = True
        for syn in wn.synsets(word):
          for l in syn.lemmas():
            string = l.name()
            if re.findall('[^A-Za-z]+', string) : break
            all_words[l.name()] = 1
    wd_list = []
    for w in tqdm(list(all_words.keys())) :
        wd_list = wd_list + transform_word_from_wordnet(w)
    
    final_wd_dict = { f : True for f in wd_list}
    with open(save_dir + '/final_wd_dict.pkl', 'wb') as f: pickle.dump(final_wd_dict, f)
else :
    with open(save_dir + '/final_wd_dict.pkl', 'rb') as f: final_wd_dict = pickle.load(f)


# ++++++++ Top 20 words

def get_top_n_words(corpus, n=100):
    corpus = [c.lower() for c in corpus]
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items() if word in final_wd_dict.keys()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return {w[0] : 1 for w in words_freq[:n]}

# ++++++++ Train/Test split
count = 1000

def iterate_by_label(c, df, count) :
    df_hts     = hts[hts.label == c]
    df_nacis   = nacis[nacis.label == c]

    arr = []
    if not df_hts.empty: arr.append(df_hts[['label', 'text']])
    if not df_nacis.empty: arr.append(df_nacis[['label', 'text']])

    this_class = get_top_n_words(list(df['text']))

    if df.shape[0] > 20 :
       df_sampled = df.sample(frac=min(count/len(df), 1))
       train_df   = df_sampled.sample(frac=0.8)
      
       valid_df   = df_sampled.drop(train_df.index)
    else :
       train_df   = df
       valid_df   = pd.DataFrame()

    arr.append(train_df)
    acc_train_df = pd.concat(arr)

    return {'train' : acc_train_df, 'valid' : valid_df, 'words' : this_class}

recompute = False

print("setup train-val split ")

word_list = {}

if recompute :
    grouped_value_dict = result.groupby(['label']).progress_apply(lambda x: iterate_by_label(x.name, x, count)).values
    train_df  = pd.concat([g['train'] for g in grouped_value_dict])
    valid_df  = pd.concat([g['valid'] for g in grouped_value_dict])
    word_list = dict(ChainMap(*[g['words'] for g in grouped_value_dict]))
    train_df.to_pickle(save_dir + "/all_train_df_1000_txfr.pkl")
    valid_df.to_pickle(save_dir + "/all_valid_df_1000_txfr.pkl")
    with open(save_dir + '/word_list.pkl', 'wb') as f: pickle.dump(word_list, f)
else :
    train_df = pd.read_pickle(save_dir + "/all_train_df_1000_txfr.pkl")
    valid_df = pd.read_pickle(save_dir + "/all_valid_df_1000_txfr.pkl")
    with open(save_dir + '/word_list.pkl', 'rb') as f: word_list = pickle.load(f)


# ++++++++++++ Examine the size of valid_df, train_df and potential batches 

print("valid_df = {}, train_df = {} ".format( len(valid_df['text']), len(train_df['text'])))

# ++++++++++++  CountVectorizer setup

count_vector = CountVectorizer(max_features=100000)
count_vector.fit(list(hts['text']) + list(nacis['text']))

label_enc = LabelEncoder() 
label_enc.fit(list(result['label']) + list(valid_df['label'])) 
with open(save_dir + '/label_enc.pkl', 'wb') as f: pickle.dump(label_enc, f)

print("Double check result",  len(list(set(
list(result['label']) + list(valid_df['label']) + list(train_df['label'])
))))

print('DEBUG', set(list(result['label']) + list(valid_df['label']) + list(train_df['label'])) - set(all_labels)  )

recompute = False

if recompute :
   train_df = train_df.assign(enc_label = label_enc.transform(list(train_df['label'])))
   valid_df = valid_df.assign(enc_label = label_enc.transform(list(valid_df['label'])))
   
   train_csv_df = train_df.reset_index().drop(['index'], axis=1)
   test_csv_df = valid_df.reset_index().drop(['index'], axis=1)
   
   ## Labelled CSV file
   print("Save labelled csv for training ", config['train_csv'])
   train_csv_df.to_csv(config['train_csv'], index=False, header=True)  
   
   # Labelled test CSV file
   print("Save labelled csv for inference ", config['test_csv'])
   test_csv_df.to_csv(config['test_csv'], index=False, header=True)  

print("Setup tokenizers...")

unknown_word = 'unknown_word'
full_set = set(list(count_vector.vocabulary_.keys()) + list(word_list.keys()))
#full_set = set(list(count_vector.vocabulary_.keys()))

print("Number of words : (This has to be in config)", len(full_set) + 2 )

vocab = {w : i for i, w in enumerate([unknown_word, 'dumb_token'] + list(full_set))}
tokenizer = tokenizers.Tokenizer(WordLevel(vocab, unknown_word))
tokenizer.pre_tokenizer = Whitespace()

print("Use padding length ", config['padding_length'])
tokenizer.enable_padding(length=int(config['padding_length']))

# Save tokenizer
recompute = False
if recompute :
  print("Save tokenizer ", config['token_config'])
  tokenizer.save(config['token_config'])
  tokenizer = tokenizers.Tokenizer.from_file(config['token_config'])



