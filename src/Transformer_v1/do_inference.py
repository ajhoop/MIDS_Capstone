#!/usr/bin/env python3
import IPython
import sys  
import pandas as pd
import pickle
sys.path.insert(0, '.')
import TransformerModel
import json
import re
from sklearn.metrics import accuracy_score

print("Using configuration file : config.json")
config = json.loads(re.sub(r'#.*?\n', '', open('config.json', 'r').read()))

df_val   = pd.read_csv(config['test_csv'], dtype={'label': str, 'text' : str, 'Index' : int})

predict = TransformerModel.Predict()
predict.setup_dataloader(df_val)

#df_val['pred_label'], df_val['cos_sim'] = predict.do_inference()
prob = predict.do_inference(method = 'prob')

df_val['first'], df_val['second'], df_val['third']  = list(prob[0][:,0]), list(prob[0][:,1]), list(prob[0][:,2])
print('Accuracy : first (probablity)', accuracy_score(df_val['label'], df_val['first']))
print('Accuracy : first + second ', accuracy_score(df_val['label'], df_val['first']) + accuracy_score(df_val['label'], df_val['second']))
print('Accuracy : first + second + third ', (accuracy_score(df_val['label'], df_val['first']) + 
                                             accuracy_score(df_val['label'], df_val['second']) +
                                             accuracy_score(df_val['label'], df_val['third']))
     )

print("\n>>> Confusion Matrix : \n")
y_true = list(df_val['label'])
y_pred = list(df_val['first'])
predict.print_confusion_matrix(y_true, y_pred)
print("\n\n\n")
