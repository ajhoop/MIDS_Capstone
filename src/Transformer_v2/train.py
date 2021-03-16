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

import pickle
import random
import numpy as np

import wandb
from pytorch_lightning.loggers import WandbLogger

from tokenizers import Tokenizer
from transformers import AdamW

import sys  
sys.path.insert(0, '.')
import TransformerModel
import json
import re

import pandas as pd

from tqdm import tqdm

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --------- Data module -------------------
class DataModule(LightningDataModule):

    def __init__(
        self,
        num_workers:   int = 16,
        seed:          int = 42,
        batch_size:    int = 32,
        fast_dev_run : bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.num_workers   = num_workers
        self.seed          = seed
        self.dataset_train = ...
        self.dataset_val   = ...
        self.stage         =  'setup'
        self.fast_dev_run  = fast_dev_run

        print("Using configuration file : config.json")
        self.config = json.loads(re.sub(r'#.*?\n', '', open('config.json', 'r').read()))
        self.batch_size    = int(self.config['batch_size'])

    # --------- Read the configuration -------------------
    def prepare_data(self):
        self.tokenizer = Tokenizer.from_file(self.config['token_config'])
        self.csv_train = pd.read_csv(self.config['train_csv'], dtype={'label': str, 'text' : str}).dropna()
        self.csv_val   = pd.read_csv(self.config['test_csv'], dtype={'label': str, 'text' : str}).dropna()
        self.padding_length = int(self.config['padding_length'])

    # --------- Encode the data into a dictionary. 
    # --------- Online encoding will slow down the CPU

        def encode_data(df) : 
            enc_dict = {}
            for i, (label, text, enc_label) in enumerate(zip(list(df.label), list(df.text), list(df.enc_label))) :
                enc = self.tokenizer.encode(text)
                ids = np.array(enc.ids[:self.padding_length])
                ids = np.vectorize(lambda x : 1 if not x else x)(ids)
                enc_dict[i] = { 'x' : ids.copy(), 'y' : enc_label, 'label' : label }
            return enc_dict

        if (self.fast_dev_run) :
          self.train_dict = encode_data(self.csv_train.head(1024))  
          self.val_dict = encode_data(self.csv_val.head(1024))  
        else :
          print('Encoding training data' )
          self.train_dict = encode_data(self.csv_train)  
          print('Encoding val data' )
          self.val_dict = encode_data(self.csv_val)  

    def do_setup(self, stage):
        #IPython.embed(); exit(1)
        """Split the train and valid dataset"""
        if (stage == 'val' and self.stage == 'train')  : return
        if (stage == 'val' and self.stage == 'setup')  : self.stage = 'train'  

        # ------ Dataset definitions - returns one sample
        class Dataset(torch.utils.data.Dataset):

              def __init__(self, data):
                    self.data = data
    
              def __len__(self):
                    return len(self.data)
 
              def __getitem__(self, index):
                    x = self.data[index]['x']
                    random.shuffle(x)
                    return  { 
                             'x'      :  torch.from_numpy(x),
                             'mask'   : (torch.from_numpy(np.array(x)) == 0),
                             'y'      :  torch.tensor(self.data[index]['y']),
                             'label'  : self.data[index]['label']
                    }

        self.dataset_train = Dataset(self.train_dict)
        self.dataset_val = Dataset(self.val_dict)

    # ------ Train dataloader 
    def train_dataloader(self):
        self.do_setup('train')

        loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return loader

    # ------ Validation dataloader
    def val_dataloader(self):
        self.do_setup('val')
        
        loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return loader

# --------- Main -------------------
class HTSClassifier(pl.LightningModule):
    """ Main classifier """

    def __init__(self, learning_rate=1e-3, config=None):
        super().__init__()
        self.save_hyperparameters()

        self.linear1 = torch.nn.Linear(in_features=2048, out_features=2048)
        self.bn1 = torch.nn.BatchNorm1d(num_features=2048)
        self.linear2 = torch.nn.Linear(in_features=2048, out_features=5386)
        self.bn2 = torch.nn.BatchNorm1d(num_features=5386)
        self.dropout = torch.nn.Dropout(p=0.001)

        self.valid_acc_1 = pl.metrics.Accuracy(compute_on_step=False)
        self.valid_acc_2 = pl.metrics.Accuracy(compute_on_step=False)
        self.valid_acc_3 = pl.metrics.Accuracy(compute_on_step=False)
        self.valid_acc_4 = pl.metrics.Accuracy(compute_on_step=False)
        self.valid_acc_5 = pl.metrics.Accuracy(compute_on_step=False)


        print("Using configuration file : config.json")
        self.config = json.loads(re.sub(r'#.*?\n', '', open('config.json', 'r').read()))

        self.tx           = TransformerModel.TransformerNet(
                            num_src_vocab = int(self.config['num_src_vocab']), 
                            embedding_dim = int(self.config['embedding_dim']), 
                            hidden_size   = int(self.config['hidden_size']), 
                            nheads        = int(self.config['nheads']), 
                            n_layers      = int(self.config['n_layers']), 
                            max_src_len   = int(self.config['padding_length']), 
                            dropout       = float(self.config['dropout']))

    def forward(self, id_one, att_one):

        # Custom transformer.
        y = self.tx(id_one, att_one)

        # Concatenate vectors instead of adding them up. 
        y = torch.squeeze(y.permute(1, 0, 2).reshape(y.size(1), 1, -1), 1)

        y = F.relu(self.bn1(self.linear1(y)))
        y = self.dropout(y)

        y = self.bn2(self.linear2(y))

        return y

    def training_step(self, batch, batch_idx):
        y = self.forward(batch['x'], batch['mask'])
        loss = F.cross_entropy(y, batch['y'])

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('learning_rate', self.opt.param_groups[0]['lr'], logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        #IPython.embed(); exit(1)
        y = self.forward(batch['x'], batch['mask'])
        loss = F.cross_entropy(y, batch['y'])

        logits = torch.softmax(y, dim=1)
        #predictions = torch.argmax(logits, dim=1)
        sorted_prob, indices = torch.sort(logits, descending=True)

        predictions_1 = indices[:,0]
        predictions_2 = indices[:,1]
        predictions_3 = indices[:,2]
        predictions_4 = indices[:,3]
        predictions_5 = indices[:,4]

        self.valid_acc_1(predictions_1, batch['y'])
        self.valid_acc_2(predictions_2, batch['y'])
        self.valid_acc_3(predictions_3, batch['y'])
        self.valid_acc_4(predictions_4, batch['y'])
        self.valid_acc_5(predictions_5, batch['y'])

        #IPython.embed(); exit(1)

        return loss

    def validation_epoch_end(self, outputs):
        loss = torch.mean(torch.stack([o for o in outputs]))
        self.log('val_loss_epoch', loss)

        top1 = self.valid_acc_1.compute()
        top2 = self.valid_acc_2.compute()
        top3 = self.valid_acc_3.compute()
        top4 = self.valid_acc_4.compute()
        top5 = self.valid_acc_5.compute()

        self.log('valid_acc_top',  top1)
        self.log('valid_acc_top5', top1 + top2 + top3 + top4 + top5)

    def configure_optimizers(self):

        #IPython.embed(); exit(1)
        self.opt = AdamW(self.parameters(), lr=0.001)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, patience=5, factor=0.65, verbose=True)

        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            'monitor': 'val_loss_epoch'
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
        parser.add_argument('--learning_rate', type=float, default=0.001)
        return parser

## --------- Customize trainer -------------------
#class SubTrainer(pl.Trainer):
#    def save_checkpoint(self, dirpath):
#        if self.is_global_zero:
#            # https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-entire-model
#            #IPython.embed(); exit(1)
#            torch.save(self.get_model().state_dict(), dirpath)


def cli_main():


    print("Using configuration file : config.json")
    config = json.loads(re.sub(r'#.*?\n', '', open('config.json', 'r').read()))

    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = HTSClassifier.add_model_specific_args(parser)
    parser = DataModule.add_argparse_args(parser)
    parser.add_argument('--run_name', type=str, default=config['lm_run_name'])
    parser.add_argument('--project', type=str, default=config['lm_project'])
    parser.add_argument('--group', type=str, default=config['lm_group'])
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dm = DataModule.from_argparse_args(args)

    # ------------
    # model
    # ------------
    model = HTSClassifier(args.learning_rate, config)

    # ------------
    # training
    # ------------
    wandb_logger = WandbLogger(name = args.run_name, project=args.project, group=args.group, save_dir=config['wandb_dir'])

    trainer = pl.Trainer.from_argparse_args(args,
      default_root_dir = config['pl_root_dir'],
      checkpoint_callback=False,
      gpus = (1 if torch.cuda.is_available() else 0),
      max_epochs = 40,
      logger = wandb_logger,
    )
    trainer.fit(model, datamodule=dm)
    # https://pytorch-lightning.readthedocs.io/en/stable/weights_loading.html
    trainer.save_checkpoint(config['lm_save_file'])

    # ------------
    # testing
    # ------------
    # todo: without passing model it fails for missing best weights
    # MisconfigurationException, 'ckpt_path is "best", but ModelCheckpoint is not configured to save the best model.'
    result = trainer.test(model, datamodule=dm)
    pprint(result)


if __name__ == '__main__':
    cli_main()
    wandb.finish()
