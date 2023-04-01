import json
import os
import pickle

import numpy as np
import pandas as pd
from utils import *

ligand_pos_df = pd.read_csv("data/mhc_ligand_pos50.csv",index_col=None,header=0)
ligand_pos_df.loc[:,"label"] = 1

neg_dir = "data/Mass_neg"
neg_files = os.listdir(neg_dir)

neg_df = pd.DataFrame(columns=["Peptide", "Allele"])
for f_i in neg_files:
    allel_neg_df = pd.read_csv("data/Mass_neg/"+f_i,index_col=None,header=0)
    neg_df = pd.concat([neg_df,allel_neg_df])
neg_df.to_csv("data/mhc_ligand_neg50.csv",index=False)

neg_df.loc[:,"label"] = 0

all_df = pd.concat([ligand_pos_df,neg_df])
all_df.to_csv("data/mhc_ligand_pos_neg50.csv",index=False)

