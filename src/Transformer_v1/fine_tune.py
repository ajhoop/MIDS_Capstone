#!/usr/bin/env python3

import IPython
from argparse import ArgumentParser
from pprint import pprint

import torch
from torch.nn import functional as F

import pytorch_lightning as pl


import platform
from typing import Optional

from torch.utils.data import DataLoader, random_split

from pytorch_lightning import LightningDataModule

import pandas as pd
from tokenizers import Tokenizer
import numpy as np
from tqdm.contrib.concurrent import process_map
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import wandb
from pytorch_lightning.loggers import WandbLogger

from transformers import AdamW


import sys  
sys.path.insert(0, '.')
import TransformerModel

import json
import re
import pickle
from tqdm import tqdm
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize


# --------- Data module -------------------
class DataModule(LightningDataModule):

    def __init__(
        self,
        num_workers: int = 16,
        seed: int = 42,
        batch_size: int = 768,
        train_samples: str = "csv/ag_news_train.csv",
        debug: bool = False,
        sample_count: int = 1500,     # How much to sample from orig data

        full_test_percent: int = 20,  # % of cartisian used as val set.
                                      # rest is used as training set.

        train_size : int  = 768 * 5,    # How much to sample from 100 - full_test_percent
        test_size : int  = 768 * 1,      # How much to sample from full_test_percent
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        self.num_workers = num_workers
        self.seed = seed
        self.batch_size = batch_size

        self.train_ds          = None
        self.test_ds           = None
        self.debug             = debug
        self.train_index_x     = []
        self.sample_count      = sample_count
        self.full_test_percent = full_test_percent
        self.train_size    = train_size
        self.test_size     = test_size
        self.current_coverage = 0
        self.stage =  'setup'

        print("Using configuration file : config.json")
        self.config = json.loads(re.sub(r'#.*?\n', '', open('config.json', 'r').read()))

        self.train_samples = self.config['train_csv']

        self.group_size     = int(self.config['group_size'])
        self.padding_length = int(self.config['padding_length'])

        #IPython.embed(); exit(1)

    def prepare_data(self):
        self.csv_train = pd.read_csv(self.train_samples, dtype={'label': str, 'text' : str, 'Index' : int}).dropna()
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

    def parallize_map(self, df, workers) :
         global _transform_one  # Promote to top
         def _transform_one(x):
             x['label'] = (x.apply(lambda df : 0.9999 if df['label_x'] == df['label_y'] else 0.0001, axis=1))
             return x
    
         X_copy = df.copy()
         data_split = np.array_split(X_copy, workers)
         return pd.concat(process_map(_transform_one, data_split, max_workers=workers))
    
    def get_product(self, chunk, num_workers=12):
        product = (
            chunk
            .drop(['text'], axis=1)
            .assign(key=1)
            .merge(chunk.drop(['text'], axis=1).assign(key=1), on="key")
            .drop("key", axis=1)
        )
        product = product[product.Index_x != product.Index_y]
        product = self.parallize_map(product, workers=num_workers)

        return (product.drop(['label_x', 'label_y'], axis=1))

    def generate_ds (self, stage):
        if (stage == 'val' and self.stage == 'train')  : return
        if (stage == 'val' and self.stage == 'setup')  : self.stage = 'train'  

        class Dataset(torch.utils.data.Dataset):

          def __init__(self, x, y, label, enc, padding_len):
                'Initialization'
                self.x, self.y, self.label, self.enc, self.padding_len = (list(x), 
                                                                          list(y), 
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
                y_dict = self.y[index]
                label  = self.label[index]

                # FIXME
                label_x, size_x, group_x = (self.enc[x_dict]['label'], self.enc[x_dict]['size'], self.enc[x_dict]['grouped_enc'])
                label_y, size_y, group_y = (self.enc[y_dict]['label'], self.enc[y_dict]['size'], self.enc[y_dict]['grouped_enc'])


                return  {
                            'label'             : torch.from_numpy(np.array([label])).to(torch.float),
                            'size_x'            : size_x,
                            'label_x'           : label_x,
                            'group_x'           : group_x,
                            'size_y'            : size_y,
                            'label_y'           : label_y,
                            'group_y'           : group_y
                        }

        #replace_val = True
        replace_val = False

        if self.debug :
          cartesian = self.get_product(self.csv_train.sample(100, replace=replace_val))
        else :
          sampled_csv_train = self.csv_train[~self.csv_train['Index'].isin(self.train_index_x)]

          if len(sampled_csv_train.index) <= self.sample_count :
            #print('sampled_csv_train.index ', len(sampled_csv_train.index), 'self.sample_count ', self.sample_count)
            extra_sample = self.csv_train.sample(self.sample_count - len(sampled_csv_train.index) + 1, replace=replace_val)
            sampled = pd.concat([sampled_csv_train, extra_sample])
            self.train_index_x = []
          else :
            sampled = sampled_csv_train.sample(self.sample_count, replace=replace_val)

          cartesian = self.get_product(sampled)

        # ---- Define train and test split, locally
        test  = cartesian.sample(frac=self.full_test_percent/100)
        train = cartesian.drop(test.index)

        test, train = test.sample(min(self.test_size, len(test.index))), train.sample(min(self.train_size, len(train.index)))
        self.train_index_x = self.train_index_x + list(set(train.Index_x))
        self.current_coverage = (len(self.train_index_x)/len(self.csv_train.index)) * 100

        self.test_ds  = Dataset(test['Index_x'], test['Index_y'], test['label'], self.encoded_dict, self.config['padding_length'])
        self.train_ds = Dataset(train['Index_x'], train['Index_y'], train['label'], self.encoded_dict, self.config['padding_length'])
        #IPython.embed(); exit(1)

    def train_dataloader(self):
        self.generate_ds("train")

        loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):

        self.generate_ds("val")
        loader = DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser.add_argument('--debug', type=bool, default=False)
        parser.add_argument('--sample_count', type=int, default=1000)
        parser.add_argument('--full_test_percent', type=int, default=20)
        return parser

# --------- Main -------------------
class LitClassifier(pl.LightningModule):
    def __init__(self, learning_rate=0.00001, config=None):
        super().__init__()
        self.save_hyperparameters()
        self.loss          = torch.nn.MSELoss()

        self.model      = TransformerModel.TransformerNet(
                            num_src_vocab = int(config['num_src_vocab']), 
                            embedding_dim = int(config['embedding_dim']), 
                            hidden_size   = int(config['hidden_size']), 
                            nheads        = int(config['nheads']), 
                            n_layers      = int(config['n_layers']), 
                            max_src_len   = int(config['padding_length']), 
                            dropout       = float(config['dropout']))

        #IPython.embed(); exit(1)
        self.group_size     = int(config['group_size'])
        self.padding_length = int(config['padding_length'])
        self.embedding_dim  = int(config['embedding_dim'])
        self.batch_size     = int(config['rf_batch_size'])
        self.load_state_dict(torch.load(config['lm_save_file']))
        self.model.train()

    def set_pointer(self, dm) : self.dm = dm

    def forward_one(self, id_one, att_one):
        x = self.model(id_one, att_one)
        return torch.squeeze(x.permute(1, 0, 2).reshape(x.size(1), 1, -1), 1)

    # https://discuss.pytorch.org/t/conditional-computation-that-saves-computation/14878/15
    def forward_one_arm(self, group_comb):
        x0 = torch.zeros(group_comb[0]['valid'].shape[0], self.embedding_dim * self.padding_length).to(self.device)
        x1 = torch.zeros(group_comb[0]['valid'].shape[0], self.embedding_dim * self.padding_length).to(self.device)

        outputs = []
        for i in range(self.group_size) :
      
           action_ind = torch.nonzero((group_comb[i]['valid'] == True).squeeze(), as_tuple=False).squeeze()
           nil_ind    = torch.nonzero((group_comb[i]['valid'] == False).squeeze(), as_tuple=False).squeeze()

           action_ind_ids = group_comb[i]['ids'].index_select(0, action_ind)
           action_ind_att = group_comb[i]['attention_mask'].index_select(0, action_ind)

           if action_ind.nelement() != 0 :
             action_x = self.forward_one(action_ind_ids, action_ind_att)
             x = x0.index_add_(0, action_ind, action_x)
             x1 = torch.add(x, x1)
        return x1


    def forward(self, group_x, group_y):
        x = self.forward_one_arm(group_x)
        y = self.forward_one_arm(group_y)
        return x, y

    def training_step(self, batch, batch_idx):
        #IPython.embed(); exit(1)
        x, y = self.forward(batch['group_x'], batch['group_y'])
        loss = 1000 * self.loss(torch.cosine_similarity(x, y), torch.squeeze(batch['label']))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('learning_rate', self.opt.param_groups[0]['lr'], logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self.forward(batch['group_x'], batch['group_y'])
        loss = 1000 * self.loss(torch.cosine_similarity(x, y), torch.squeeze(batch['label']))
        return loss

    def validation_epoch_end(self, outputs):
        loss = torch.mean(torch.stack([o for o in outputs]))
        #self.log('learning_rate', self.opt.param_groups[0]['lr'])
        self.log('avg_val_loss', loss)
        self.log('data_coverage', self.dm.current_coverage)


    def configure_optimizers(self):
        self.opt = AdamW(self.parameters())

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, patience=30, factor=0.5, verbose=True)

        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            'monitor': 'avg_val_loss'
        }
        return [self.opt], [scheduler]
#        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, gamma=0.1, milestones=[2], verbose=True)
#
#        scheduler = {
#            'scheduler': lr_scheduler,
#        }
#        return [self.opt], [scheduler]


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.00001)
        #parser.add_argument('--model_path', type=str, default="/scratch/models/lm.ckpt")
        return parser

# --------- Customize trainer -------------------
class SubTrainer(pl.Trainer):
    def save_checkpoint(self, dirpath):
        if self.is_global_zero:
            # https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-entire-model
            #IPython.embed(); exit(1)
            transformer = self.get_model().model
            torch.save(transformer.state_dict(), dirpath)



def cli_main():
    pl.seed_everything(1234)


    print("Using configuration file : config.json")
    config = json.loads(re.sub(r'#.*?\n', '', open('config.json', 'r').read()))

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = SubTrainer.add_argparse_args(parser)
    parser = LitClassifier.add_model_specific_args(parser)
    parser = DataModule.add_argparse_args(parser)
    parser.add_argument('--run_name', type=str, default=config['rf_run_name'])
    parser.add_argument('--project', type=str, default=config['rf_project'])
    parser.add_argument('--group', type=str, default=config['rf_group'])
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dm = DataModule.from_argparse_args(args)

    # ------------
    # model
    # ------------
    model = LitClassifier(args.learning_rate, config)
    model.set_pointer(dm)

    # ------------
    # training
    # ------------
    wandb_logger = WandbLogger(name = args.run_name, project=args.project, group=args.group)

    trainer = SubTrainer.from_argparse_args(args,
      default_root_dir = config['pl_root_dir'],
      checkpoint_callback=False,
      gpus = (1 if torch.cuda.is_available() else 0),
      max_epochs = 200,
      reload_dataloaders_every_epoch = True,
      logger = wandb_logger,
    )
    trainer.fit(model, datamodule=dm)
    trainer.save_checkpoint(config['rf_save_file'])


if __name__ == '__main__':
    cli_main()
