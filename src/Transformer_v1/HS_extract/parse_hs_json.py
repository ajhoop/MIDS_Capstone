#!/usr/bin/env python3

# Download the json file from below
# -> https://catalog.data.gov/dataset/harmonized-tariff-schedule-hts-archive
# -> https://www.usitc.gov/tata/hts/archive/index.htm

# wget https://www.usitc.gov/sites/default/files/tata/hts/hts_2021_preliminary_revision_2_json.json

# ./parse_hs_json.py  --input_hs_json  hts_2021_preliminary_revision_2_json.json  --output_csv  hts_2021_preliminary_revision_2_json.csv

import os
import sys
#from argparse import ArgumentParser
import pandas as pd
from torchvision.datasets.utils import download_url
import re
import json

config = json.loads(re.sub(r'#.*?\n', '', open('config.json', 'r').read()))

download_url(config['hts_url'], '.')

#parser = ArgumentParser(add_help=True)
#parser.add_argument('--input_hs_json', type=str, required=True, help='Input Json HS file')
#parser.add_argument('--output_csv', type=str, required=True, help='Output CSV file')


class obj:

    # constructor 
    def __init__(self, dict1):
        self.__dict__.update(dict1)
        self.child  = []
        self.id  = None
    def add_child(self, o): self.child.append(o)
    def has_child(self): 
        if len(self.child) > 0 : return True
        return False

    def get_child(self): return self.child
    def get_description(self):
        if (hasattr(self, 'description')) :
           return self.description
        return ""

def parse_and_dump(input_file) :
   json_array = json.load(open(input_file), object_hook=obj)
   
   parent   = []
   children = []
   
   cur_intent = 0
   prv_intent = 0
   prev_obj = json_array[0]
   parent.append(json_array[0])
   
   for i, cur_obj in enumerate(json_array):
      cur_obj.id = i
      prv_intent = cur_intent
   
      cur_intent = int(cur_obj.indent)
   
      if (cur_intent > prv_intent) :
        parent.append(prev_obj)
   
      if (cur_intent < prv_intent) :
        [parent.pop() for i in range(prv_intent - max(cur_intent, 1))]
   
      if (cur_intent != 0) :
        parent[-1].add_child(cur_obj)
   
      prev_obj   = cur_obj
   
   new_json_array = list(filter(lambda x : int(x.indent) == 0, json_array))
   
   #print(json.dumps( new_json_array, default=lambda o: o.__dict__, sort_keys=True, indent=2))

   rows_list = []
   
   def get_description(o, s) :

       description = o.get_description()
       description = description.replace(':',' ')
       s["level_{}".format(o.indent)] = o.get_description()
   
       if o.htsno and not o.htsno.isspace() :
          #d = { 'indent' : o.indent, 'htsno' : o.htsno, 'description' : o.get_description(), 'merged_description' : description}
          #d = { "level_{}".format(o.indent) : o.get_description(), 'htsno' : o.htsno, }
          d = s.copy()
          d['htsno'] = o.htsno

          if not o.has_child() :
            rows_list.append(d)
          #print(o.id, o.indent, ",", o.htsno, ",\"", description, "\"")
   
       if len(o.get_child()) > 0 :
         for i in o.get_child() :
           get_description(i, s)
   
   for j in new_json_array :
     get_description(j, {})
   df = pd.DataFrame(rows_list)
   col = list(df.columns)
   col.remove('htsno')
   return df[['htsno'] + col]

if __name__ == "__main__":
    #args = parser.parse_args()
    #out_file = args.output_csv
    #in_file = args.input_hs_json
    out_file = config['input_hts_data']
    in_file  = config['input_hs_json']

    if not out_file.endswith('.csv') : 
       print("\"{}\" is not a valid file name. Please give the .csv extension".format(out_file)) 
       sys.exit()

    if not os.path.isfile(in_file):
       print("\"{}\" - file does not exist please give a valid file as input".format(in_file)) 
       sys.exit()
     
    print("Parsing file.. {}".format(in_file))
    df = parse_and_dump(in_file)
    print("Writing csv file.. {}".format(out_file,))
    df.to_csv(out_file, index=True)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(df.head(10))

