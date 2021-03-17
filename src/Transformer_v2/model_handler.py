"""
https://github.com/pytorch/serve/blob/master/docs/custom_service.md
https://github.com/pytorch/serve/blob/master/examples/text_to_speech_synthesizer/waveglow_handler.py

ModelHandler defines a custom model handler.
"""

from ts.torch_handler.base_handler import BaseHandler
#from ModelForInf import HTSClassifier
from tokenizers import Tokenizer
import numpy as np
import torch
import pickle

import os

class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self.context = None
        self.initialized = False
        self.explain = False
        self.target = 0
        self.model = None

    def initialize(self, context):
        self.context = context
        model_dir    = context.system_properties.get("model_dir")
        serialized_file = context.manifest["model"]["serializedFile"]
        model_pt_path = model_dir + "/" + serialized_file
        state_dict = torch.load(model_pt_path)

        self.initialized = True

        #self.model          = HTSClassifier().eval()
        #self.model.load_state_dict(state_dict)

        #checkpoint = torch.load(state_dict)
        #self.model.load_state_dict(checkpoint['state_dict'])
        self.model          = torch.jit.load(model_pt_path)


        self.tokenizer      = Tokenizer.from_file(model_dir + "/tokenizer.json")
        self.padding_length = 64
        self.num_samples    = 5
        with open(model_dir + '/index_to_name.pkl', 'rb') as f: self.label_enc = pickle.load(f)

    def preprocess(self, data):
        preprocessed_data = data[0].get("data")
        if preprocessed_data is None: preprocessed_data = data[0].get("body")

        if isinstance(preprocessed_data, dict)  : 
          preprocessed_data = preprocessed_data['data']
        else :
          preprocessed_data = preprocessed_data.decode('utf-8') 

        print('QUERY', preprocessed_data)
        enc = self.tokenizer.encode(preprocessed_data)
        ids = np.array(enc.ids[:self.padding_length])
        ids = np.vectorize(lambda x : 1 if not x else x)(ids)
        mask  = (torch.from_numpy(np.array(ids)) == 0)
        ids = torch.from_numpy(ids)

        return ids, mask


    def inference(self, ids, mask):
        y = self.model.forward(ids, mask)
        logits = torch.softmax(y, dim=1) 
        sorted_prob, indices = torch.sort(logits, descending=True)
        sorted_prob, indices = sorted_prob.detach(), indices.detach()
        #print('INDICES', indices)
        #print('PROB', sorted_prob)
        return sorted_prob, indices  

    def postprocess(self, sorted_prob, indices):
        indices = np.vectorize(self.label_enc.get)(indices[0].numpy()[:self.num_samples])
        sorted_prob = sorted_prob[0].numpy()[:self.num_samples]
        return [[{i:s} for i, s in zip(indices.tolist(), sorted_prob.tolist())]] 

    def handle(self, data, context):
        ids, mask = self.preprocess(data)
        sorted_prob, indices = self.inference(ids.reshape(1, self.padding_length), 
                                              mask.reshape(1, self.padding_length))
        return self.postprocess(sorted_prob, indices)
