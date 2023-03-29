import json
import os
import pickle

import numpy as np
import pandas as pd
from utils import *

ligand_pos_df = pd.read_csv("data/mhc_ligand_pos.csv",index=None,header=0)

neg_dir = "data/Mass_neg"
neg_files = os.listdir(neg_dir)
for f_i in neg_files:
    neg_df = pd.read_csv("data/Mass_neg/"+f_i,index=None,header=None)
