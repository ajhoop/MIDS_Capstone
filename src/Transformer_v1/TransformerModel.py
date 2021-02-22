#!/usr/bin/env python3
import IPython
import torch
import numpy as np
import json
from tokenizers import Tokenizer
from tqdm import tqdm
import re
import pickle
import pandas as pd
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize


from typing import List, Optional
from termcolor import colored
from sklearn.metrics import confusion_matrix

# --------- Basic TX -------------------
## source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #IPython.embed(); exit(1)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerNet(torch.nn.Module):
  def __init__(self, num_src_vocab, embedding_dim, hidden_size, nheads, n_layers, max_src_len, dropout):
    super(TransformerNet, self).__init__()
    # embedding layers
    self.enc_embedding = torch.nn.Embedding(num_src_vocab, embedding_dim)

    # positional encoding layers
    self.enc_pe = PositionalEncoding(embedding_dim, max_len = max_src_len)

    # encoder/decoder layers
    enc_layer = torch.nn.TransformerEncoderLayer(embedding_dim, nheads, hidden_size, dropout)
    self.encoder = torch.nn.TransformerEncoder(enc_layer, num_layers = n_layers)

  # https://github.com/pytorch/pytorch/blob/11f1014c05b902d3eef0fe01a7c432f818c2bdfe/torch/nn/functional.py#L4110
  # src: (S, N, E)
  # src_mask: (S, S)
  # src_key_padding_mask: (N, S)

  def forward(self, src, src_key_padding_mask=None):
    src = self.enc_embedding(src).permute(1, 0, 2)
    src = self.enc_pe(src)

    # Pass the mask input through the encoder layers in turn
    memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
    return memory


# --------- Convert samples into trained embeddings -------------------

class Misc():

    def __init__(self, 
                 batch_size = 1024, 
                 ) :

        self.device     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print("Using configuration file : config.json")
        self.config = json.loads(re.sub(r'#.*?\n', '', open('config.json', 'r').read()))

        self.model      = TransformerNet(
                            num_src_vocab = int(self.config['num_src_vocab']), 
                            embedding_dim = int(self.config['embedding_dim']), 
                            hidden_size   = int(self.config['hidden_size']), 
                            nheads        = int(self.config['nheads']), 
                            n_layers      = int(self.config['n_layers']), 
                            max_src_len   = int(self.config['padding_length']), 
                            dropout       = float(self.config['dropout']))

        self.model.load_state_dict(torch.load(self.config['rf_save_file']))
        self.model.eval()
        self.model.to(self.device)

        self.max_length      = int(self.config['padding_length'])
        self.tokenizer       = Tokenizer.from_file(self.config['token_config'])
        self.batch_size      = batch_size
        self.padding_length = int(self.config['padding_length'])
        self.embedding_dim  = int(self.config['embedding_dim'])

        self.padding_length = int(self.config['padding_length'])
        self.group_size     = int(self.config['group_size'])
        self.num_workers    = 4
     
    def prepare_data(self, df_in):

        self.csv_train = df_in.dropna()
        tokenizer     = Tokenizer.from_file(self.config['token_config'])
        self.encoded_dict = {}

        def split_chunks(l, pl = self.padding_length):
            d = len(l)
            for i in range(d):
                if (i % pl == 0) :  
                    yield " ".join(l[i:min(i+pl-1, d)])

        empty_list = [{
          'ids'            : torch.from_numpy(np.array([0] * self.padding_length)), 
          'attention_mask' : (torch.from_numpy(np.array([0] * self.padding_length)) == 0),
          #'item'           : '',
          'valid'          : False
        }] * self.group_size
        
        for i, label, text in zip(list(self.csv_train.Index), list(self.csv_train.label), list(self.csv_train.text)) :
            toks = word_tokenize(text)
            grouped_enc = []
            for item in list(split_chunks(toks)) : 
                enc = tokenizer.encode(item)
                grouped_enc.append({
                                     'ids'            : torch.from_numpy(np.array(enc.ids)[:self.padding_length]), 
                                     'attention_mask' : (torch.from_numpy(np.array(enc.attention_mask)[:self.padding_length]) == 0), 
                                     #'item'           : item,
                                     'valid'          : True
                                   })
                # a = (a + N * [''])[:N]
                grouped_enc = (grouped_enc + empty_list)[:self.group_size] 

            self.encoded_dict[i] = {'label' : label, 'size' :len(grouped_enc), 'grouped_enc' : grouped_enc}

        class Dataset(torch.utils.data.Dataset):

          def __init__(self, x, label, enc, padding_len):
                'Initialization'
                self.x, self.label, self.enc, self.padding_len = (list(x), 
                                                                  list(label), 
                                                                  enc, 
                                                                  int(padding_len))

          def __len__(self):
                'Denotes torche total number of samples'
                return len(self.label)

          def __getitem__(self, index):
                'Generates one sample of data'
                # Select sample
                x_dict = self.x[index]

                group_x = self.enc[x_dict]['grouped_enc']

                return  { 'group_x'           : group_x }
        self.train_ds = Dataset(
                                 self.csv_train['Index'], 
                                 self.csv_train['label'], 
                                 self.encoded_dict, 
                                 self.config['padding_length'])
    
    def dataloader(self, df):
        self.prepare_data(df)
        loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return loader

    def forward_one(self, id_one, att_one):
        x = self.model(id_one, att_one)
        return torch.squeeze(x.permute(1, 0, 2).reshape(x.size(1), 1, -1), 1)


    def forward_one_arm(self, group_comb):
        x0 = torch.zeros(group_comb[0]['valid'].shape[0], self.embedding_dim * self.padding_length).to(self.device)
        x1 = torch.zeros(group_comb[0]['valid'].shape[0], self.embedding_dim * self.padding_length).to(self.device)

        outputs = []
        for i in range(self.group_size) :
      
           action_ind = torch.nonzero((group_comb[i]['valid'] == True).squeeze(), as_tuple=False).squeeze()
           #nil_ind    = torch.nonzero((group_comb[i]['valid'] == False).squeeze(), as_tuple=False).squeeze()

           action_ind_ids = group_comb[i]['ids'].index_select(0, action_ind)
           action_ind_att = group_comb[i]['attention_mask'].index_select(0, action_ind)

           if action_ind.nelement() != 0 :
             action_x = self.forward_one(action_ind_ids, action_ind_att)
             x = x0.index_add_(0, action_ind, action_x)
             x1 = torch.add(x, x1)
        return x1

    def move_to(self, obj, device) :
        if torch.is_tensor(obj):
          return obj.to(device)
        elif isinstance(obj, dict):
          res = {}
          for k, v in obj.items():
            res[k] = self.move_to(v, device)
          return res
        elif isinstance(obj, list):
          res = []
          for v in obj:
            res.append(self.move_to(v, device))
          return res
        else:
          raise TypeError("Invalid type for move_to")

    
    def get_embedding(self, df):
        ds = self.dataloader(df)
        
        accumulate = []
        for i, g in enumerate(tqdm(ds)):
            group = self.move_to(g['group_x'], self.device)
            #print(group[0]['ids'].shape)
            #IPython.embed()
         
            x = self.forward_one_arm(group)
            #IPython.embed(); exit(1)
            output_vector = x.detach()
            accumulate.append(output_vector)
        return torch.cat(accumulate).to('cpu')

# ------  The predict class. For classification, each datapoint need 
# to be compared against each centroid. Better do this operation 
# in GPU (if available)using vector operations for saving time.

class Predict():

    def __init__(self, 
                 df_train   = None, 
                 batch_size = 1024, 
                 ) :

        self.device          = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print("Using configuration file : config.json")
        config = json.loads(re.sub(r'#.*?\n', '', open('config.json', 'r').read()))

        self.model      = TransformerNet(
                            num_src_vocab = int(config['num_src_vocab']), 
                            embedding_dim = int(config['embedding_dim']), 
                            hidden_size   = int(config['hidden_size']), 
                            nheads        = int(config['nheads']), 
                            n_layers      = int(config['n_layers']), 
                            max_src_len   = int(config['padding_length']), 
                            dropout       = float(config['dropout']))

        self.model.load_state_dict(torch.load(config['rf_save_file']))
        self.model.eval()
        self.model.to(self.device)


        self.dim             = int(config['embedding_dim']) * int(config['padding_length'])
        self.embedding_dim   = int(config['embedding_dim'])
        self.padding_length  = int(config['padding_length'])

        self.df_train        = df_train
        self.max_length      = int(config['padding_length'])
        self.tokenizer       = Tokenizer.from_file(config['token_config'])
        self.batch_size      = batch_size

        self.predict_group_size = int(config['predict_group_size'])

        # Basic pytorch similarity function
        self.cos     = torch.nn.CosineSimilarity(dim=1)
        self.softmax = torch.nn.Softmax(dim=1)

    def process_centroids(self):
        # Group the embeddings together for getting centroids.
        df_train_summed = self.df_train.groupby('label').apply(lambda x: x['embedding'].values).to_frame()

        # Get the mean of the points for getting centroids.
        def create_dict(r):
            return np.concatenate([[np.array(i)] for i in r]).mean(axis=0)

        # Get the list of centroids and list of labels (names) in df_final and df_index.
        self.df_final = np.concatenate([[create_dict(row[0])] for index, row in df_train_summed.iterrows()])
        self.df_index = [index for index, row in df_train_summed.iterrows()]
 
#    def prepare_data(self, df):
#        class Dataset(torch.utils.data.Dataset):
#
#              def __init__(self, tokenizer, training_data, padding_length):
#                    self.tokenizer      = tokenizer
#                    self.training_data  = training_data
#                    self.padding_length = padding_length
#    
#              def __len__(self):
#                    return len(self.training_data)
# 
#              def __getitem__(self, index):
#                    enc = self.tokenizer.encode(self.training_data[index])
#                    return  { 
#                             'ids'            : torch.Tensor(enc.ids[:self.padding_length]).to(torch.int64), 
#                             'attention_mask' : (torch.from_numpy(np.array(enc.attention_mask)[:self.padding_length]) == 0),
#                    }
#
#        return Dataset(self.tokenizer, df, self.max_length)
#
#    
#    def dataloader(self, df):
#        return torch.utils.data.DataLoader(
#                self.prepare_data(df),
#                batch_size=self.batch_size,
#                num_workers=self.num_workers,
#                pin_memory=True,
#               )
    
    # FIXME
#    def get_embedding(self, df):
#        ds = self.dataloader(df)
#        accumulate = []
#        for i, k in enumerate(tqdm(ds)):
#            input_ids = k['ids']
#            input_ids = input_ids.to(self.device)
#            x = self.model(input_ids)
#            output_vector = torch.squeeze(x.permute(1, 0, 2).reshape(x.size(1), 1, -1), 1)
#            output_vector = output_vector.detach()
#            accumulate.append(output_vector)
#        #return torch.cat(accumulate).to('cpu')
#        return torch.cat(accumulate)
    def move_to(self, obj, device) :
        if torch.is_tensor(obj):
          return obj.to(device)
        elif isinstance(obj, dict):
          res = {}
          for k, v in obj.items():
            res[k] = self.move_to(v, device)
          return res
        elif isinstance(obj, list):
          res = []
          for v in obj:
            res.append(self.move_to(v, device))
          return res
        else:
          raise TypeError("Invalid type for move_to")

    def forward_one(self, id_one, att_one):
        x = self.model(id_one, att_one)
        return torch.squeeze(x.permute(1, 0, 2).reshape(x.size(1), 1, -1), 1)


    def forward_one_arm(self, group_comb):
        x0 = torch.zeros(group_comb['ids'].shape[0], self.embedding_dim * self.padding_length).to(self.device)
        x1 = torch.zeros(group_comb['ids'].shape[0], self.embedding_dim * self.padding_length).to(self.device)

        outputs = []
        for i in range(self.predict_group_size) :
      
           action_ind = torch.nonzero((group_comb['valid'] == True).squeeze(), as_tuple=False).squeeze()
           #nil_ind    = torch.nonzero((group_comb['valid'] == False).squeeze(), as_tuple=False).squeeze()

           action_ind_ids = group_comb['ids'].index_select(0, action_ind)
           action_ind_att = group_comb['attention_mask'].index_select(0, action_ind)

           if action_ind.nelement() != 0 :
             action_x = self.forward_one(action_ind_ids, action_ind_att)
             x = x0.index_add_(0, action_ind, action_x)
             x1 = torch.add(x, x1)
        return x1

    def get_embedding(self, group):
        #IPython.embed(); exit(1)
        group = self.move_to(group, self.device)
        x = self.forward_one_arm(group)
        output_vector = x.detach()
        return output_vector

    def setup_dataloader(self, df_in):

        class Dataset(torch.utils.data.Dataset):

              def __init__(self, tokenizer, training_data, padding_length):
                    self.tokenizer      = tokenizer
                    self.training_data  = training_data
                    self.padding_length = padding_length
    
              def __len__(self):
                    return len(self.training_data)
 
              def __getitem__(self, index):
                    enc = self.tokenizer.encode(self.training_data[index])
                    return  { 
                             'ids'            : torch.Tensor(enc.ids[:self.padding_length]).to(torch.int64), 
                             'attention_mask' : (torch.from_numpy(np.array(enc.attention_mask)[:self.padding_length]) == 0),
                             'valid'          : True,
                    }

#        class Dataset(torch.utils.data.Dataset):
#            
#          def __init__(self, list_IDs):
#                'Initialization'
#                self.list_IDs = list_IDs
#        
#          def __len__(self):
#                'Denotes the total number of samples'
#                return len(self.list_IDs)
#        
#          def __getitem__(self, index):
#                'Generates one sample of data'
#                # Select sample
#                X = self.list_IDs[index]
#                return X

        training_set = Dataset(self.tokenizer, [l.lower() for l in list(df_in['text'])], self.max_length)

        self.training_generator = torch.utils.data.DataLoader(
                training_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=12,
                pin_memory=True,
               )


    def compute(self, embedding) :
        if (embedding.shape[1] != self.dim) : print('embedding : Dim does not match. Expected : ', embedding.shape[1]) # For sanity check
        if (self.cen.shape[1] != self.dim) : print('self.cen : Dim does not match : ', self.cen.shape[1]) # For sanity check
        a = torch.repeat_interleave(embedding, repeats=self.cen.shape[0], dim=0)
        b = self.cen.repeat(embedding.shape[0], 1)
        output = self.cos(a, b)
        rsh = torch.reshape(output, (embedding.shape[0], self.cen.shape[0]))
        amax = torch.argmax(rsh, dim=1)
        maxval = rsh[torch.arange(rsh.shape[0]), amax]
        
        # Returns maximum cosine similarity and max cos values that yielded 
        # the similarity for the batch.
        return amax.to('cpu').numpy(), maxval.to('cpu').numpy()


    def compute_prob(self, embedding) :
        if (embedding.shape[1] != self.dim) : print('embedding : Dim does not match. Expected : ', embedding.shape[1]) # For sanity check
        if (self.cen.shape[1] != self.dim) : print('self.cen : Dim does not match : ', self.cen.shape[1]) # For sanity check

        a = torch.repeat_interleave(embedding, repeats=self.cen.shape[0], dim=0)
        b = self.cen.repeat(embedding.shape[0], 1)
        output = self.cos(a, b)
        rsh = torch.reshape(output, (embedding.shape[0], self.cen.shape[0]))

        sorted_cos, indices = torch.sort(rsh, descending=True)
        sm = self.softmax(sorted_cos)

        return indices.to('cpu').numpy(), sorted_cos.to('cpu').numpy(), sm.to('cpu').numpy()

    def do_inference(self, method = 'max'):

        self.cen = torch.tensor(self.df_final)
        self.cen = self.cen.to(self.device) 

        labels = np.array([])
        cos_vals = np.array([])
        softmax_vals = np.array([])
        
        # Iterate over the batch, feed to GPU.
        for i, local_batch in enumerate(tqdm(self.training_generator)) :
            #IPython.embed(); exit(1)
            sample_embedding = self.get_embedding(local_batch)
            
            if method == 'max' :
              arr, max_val = self.compute(sample_embedding)
              labels = np.append(labels, arr)
              cos_vals = np.append(cos_vals, max_val)

            if method == 'prob' :
              indices, sorted_cos, sm = self.compute_prob(sample_embedding)
              labels = np.append(labels, indices)
              cos_vals = np.append(cos_vals, sorted_cos)
              softmax_vals = np.append(softmax_vals, sm)

 
        if method == 'max' :
           return (
                   # Map the indexs back to labels
                   list(map(lambda x: self.df_index[x], list(labels.astype(int)))), 
           
                   # Return the max values that lead to inference
                   list(cos_vals)
                  )

        if method == 'prob' :

           num_classes = len(self.df_index)
           #all_labels = [self.df_index[l.astype(int)] for l in list(np.reshape(labels, (-1, num_classes)))]
           #IPython.embed(); exit(1)
           vfunc = np.vectorize(lambda t: self.df_index[t])
           #vfunc(x)

           return (
                   #np.reshape(labels.astype(int), (-1, num_classes)), 
                   vfunc(np.reshape(labels.astype(int), (-1, num_classes))), 
                   np.reshape(cos_vals, (-1, num_classes)), 
                   np.reshape(softmax_vals, (-1, num_classes))
                  )

    def print_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List] = None,
        hide_zeroes: bool = False,
        hide_diagonal: bool = False,
        hide_threshold: Optional[float] = None,
    ):
        """Print a nicely formatted confusion matrix with labelled rows and columns.
    
        Predicted labels are in the top horizontal header, true labels on the vertical header.
    
        Args:
            y_true (np.ndarray): ground truth labels
            y_pred (np.ndarray): predicted labels
            labels (Optional[List], optional): list of all labels. If None, then all labels present in the data are
                displayed. Defaults to None.
            hide_zeroes (bool, optional): replace zero-values with an empty cell. Defaults to False.
            hide_diagonal (bool, optional): replace true positives (diagonal) with empty cells. Defaults to False.
            hide_threshold (Optional[float], optional): replace values below this threshold with empty cells. Set to None
                to display all values. Defaults to None.
        """
        if labels is None:
            labels = np.unique(np.concatenate((y_true, y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        # find which fixed column width will be used for the matrix
        columnwidth = max(
            [len(str(x)) for x in labels] + [5]
        )  # 5 is the minimum column width, otherwise the longest class name
        empty_cell = ' ' * columnwidth
    
        # top-left cell of the table that indicates that top headers are predicted classes, left headers are true classes
        padding_fst_cell = (columnwidth - 3) // 2  # double-slash is int division
        fst_empty_cell = padding_fst_cell * ' ' + 't/p' + ' ' * (columnwidth - padding_fst_cell - 3)
    
        # Print header
        print('    ' + fst_empty_cell, end=' ')
        for label in labels:
            print(colored(f'{label:{columnwidth}}', 'blue'), end=' ')  # right-aligned label padded with spaces to columnwidth
    
        print()  # newline
        # Print rows
        for i, label in enumerate(labels):
            print(colored(f'    {label:{columnwidth}}', 'blue'), end=' ')  # right-aligned label padded with spaces to columnwidth
            for j in range(len(labels)):
                # cell value padded to columnwidth with spaces and displayed with 1 decimal
                cell = f'{cm[i, j]:{columnwidth}.1f}'
                if hide_zeroes:
                    cell = cell if float(cm[i, j]) != 0 else empty_cell
                if hide_diagonal:
                    cell = cell if i != j else empty_cell
                if hide_threshold:
                    cell = cell if cm[i, j] > hide_threshold else empty_cell
                if i == j : cell = colored(cell, 'red')
             
                print(cell, end=' ')
            print()
