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
from tokenizers import Tokenizer
import random
import numpy as np

import wandb
from pytorch_lightning.loggers import WandbLogger


import sys  
sys.path.insert(0, '.')
import TransformerModel
import json
import re

# --------- Data module -------------------
class DataModule(LightningDataModule):

    def __init__(
        self,
        val_split: int = 3, # 5% -> validation
        num_workers: int = 16,
        seed: int = 42,
        #batch_size: int = 1536,
        batch_size: int = 768,
        *args,
        **kwargs,
    ):
        """
        Args:
            val_split: how many of the training images to use for the validation split
            num_workers: how many workers to use for loading data
        """
        super().__init__(*args, **kwargs)

        self.dims = (1, 28, 28)
        self.val_split = val_split
        self.num_workers = num_workers
        self.seed = seed
        self.batch_size = batch_size
        self.dataset_train = ...
        self.dataset_val = ...
        self.stage =  'setup'

        print("Using configuration file : config.json")
        self.config = json.loads(re.sub(r'#.*?\n', '', open('config.json', 'r').read()))

    def prepare_data(self):
        with open(self.config['training_data_pkl'], 'rb') as f: self.training_data = pickle.load(f)
        random.shuffle(self.training_data)
        #IPython.embed(); exit(1)
        self.tokenizer = Tokenizer.from_file(self.config['token_config'])

    def do_setup(self, stage):
        #IPython.embed(); exit(1)
        """Split the train and valid dataset"""
        if (stage == 'val' and self.stage == 'train')  : return
        if (stage == 'val' and self.stage == 'setup')  : self.stage = 'train'  

        class Dataset(torch.utils.data.Dataset):

              def __init__(self, tokenizer, training_data, padding_length):
                    self.tokenizer = tokenizer
                    self.training_data = training_data
                    self.padding_length = int(padding_length)
    
              def __len__(self):
                    'Denotes the total number of samples'
                    return len(self.training_data)
 
              def mask_id(self, nonzeroind, ids) :
                  selected_id = np.random.randint(nonzeroind)
                  ids[selected_id] = 0
                  return ids

              def __getitem__(self, index):
                    'Generates one sample of data'
                    # Select sample


                    enc = self.tokenizer.encode(self.training_data[index])
                    ids = enc.ids
                    attention_mask = enc.attention_mask
                    nonzeroind = np.nonzero(ids)[0][-1]
                    orig_ids = ids.copy()
                    
                    if attention_mask[1] == 1 : ids = self.mask_id(nonzeroind, ids)
                    if attention_mask[5] == 1 : ids = self.mask_id(nonzeroind, ids)
                    if attention_mask[7] == 1 : ids = self.mask_id(nonzeroind, ids)
                    if attention_mask[10] == 1 : ids = self.mask_id(nonzeroind, ids)
                    if attention_mask[13] == 1 : ids = self.mask_id(nonzeroind, ids)
                    if attention_mask[17] == 1 : ids = self.mask_id(nonzeroind, ids)
                    if attention_mask[21] == 1 : ids = self.mask_id(nonzeroind, ids)
                    if attention_mask[25] == 1 : ids = self.mask_id(nonzeroind, ids)
                    if attention_mask[29] == 1 : ids = self.mask_id(nonzeroind, ids)

                    return  { 
                             'ids'      : torch.Tensor(ids[:self.padding_length]).to(torch.int64), 
                             'orig_ids' : torch.Tensor(orig_ids[:self.padding_length]).to(torch.int64), 
           
                             # True positions should be masked
                             'attention_mask' : (torch.from_numpy(np.array(attention_mask)[:self.padding_length]) == 0),
                             'label' : torch.from_numpy(np.array([0.99999])).to(torch.float)
                    }

        dataset = Dataset(self.tokenizer, self.training_data, self.config['padding_length'])
        train_length = len(dataset)
        val_split = int((self.val_split/100) * train_length)
        self.dataset_train, self.dataset_val = random_split(dataset, [train_length - val_split, val_split])

    def train_dataloader(self):
        self.do_setup('train')

        """train set removes a subset to use for validation"""
        loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            #drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        self.do_setup('val')
        
        """val set uses a subset of the training set for validation"""
        loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            #drop_last=True,
            pin_memory=True,
        )
        return loader

# --------- Main -------------------
class LitClassifier(pl.LightningModule):
    """ Main classifier """

    def __init__(self, learning_rate=1e-3, config=None):
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

    def forward(self, ids, orig_ids, atten):
        x = self.model(ids, atten)
        y = self.model(orig_ids, atten)
        k1 = x.permute(1, 0, 2)
        return torch.squeeze(x.permute(1, 0, 2).reshape(x.size(1), 1, -1), 1), torch.squeeze(y.permute(1, 0, 2).reshape(y.size(1), 1, -1), 1)

    def training_step(self, batch, batch_idx):
        #IPython.embed(); exit(1)
        x, y = self.forward(batch['ids'], batch['orig_ids'], batch['attention_mask'])
        loss = 1000 * self.loss(torch.cosine_similarity(x, y), torch.squeeze(batch['label']))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self.forward(batch['ids'], batch['orig_ids'], batch['attention_mask'])
        loss = 1000 * self.loss(torch.cosine_similarity(x, y), torch.squeeze(batch['label']))
        #self.log('valid_loss', loss)
        return loss

    def validation_epoch_end(self, outputs):
        loss = torch.mean(torch.stack([o for o in outputs]))
        self.log('avg_val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser

# --------- Customize trainer -------------------
class SubTrainer(pl.Trainer):
    def save_checkpoint(self, dirpath):
        if self.is_global_zero:
            # https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-entire-model
            #IPython.embed(); exit(1)
            transformer = self.get_model()
            torch.save(self.get_model().state_dict(), dirpath)


def cli_main():


    print("Using configuration file : config.json")
    config = json.loads(re.sub(r'#.*?\n', '', open('config.json', 'r').read()))

    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    #parser = pl.Trainer.add_argparse_args(parser)
    parser = SubTrainer.add_argparse_args(parser)
    parser = LitClassifier.add_model_specific_args(parser)
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
    model = LitClassifier(args.learning_rate, config)

    # ------------
    # training
    # ------------
    wandb_logger = WandbLogger(name = args.run_name, project=args.project, group=args.group)

    #trainer = pl.Trainer.from_argparse_args(args,
    trainer = SubTrainer.from_argparse_args(args,
      default_root_dir = config['pl_root_dir'],
      checkpoint_callback=False,
      gpus = (1 if torch.cuda.is_available() else 0),
      max_epochs = 2,
      reload_dataloaders_every_epoch = True,
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
