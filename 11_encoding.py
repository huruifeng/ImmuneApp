import json
import pickle

import numpy as np
import pandas as pd
from utils import *

blosum_matrix = read_blosum("data/original_data/blosum50.txt")

with open("data/allele_pseudo_seq.txt", "r") as fp:
    pseudo_seq_dict = json.load(fp)  # encode dict into JSON

binding_df = pd.read_csv("data/binding_data.csv",index_col=None,header=0)
binding_df_len14 = binding_df.loc[binding_df.peplen>=14,:]

allele_ls = binding_df.allele.tolist()
pep_ls = binding_df.peptide.tolist()
ba_ls = binding_df.affinity.tolist()
score_ls = binding_df.score.tolist()
binding_pairs = list(zip(allele_ls,pep_ls,score_ls,ba_ls))

def hla_peptide_pairs_encode(binding_pairs, allele_pseudo_seqs, blosum_matrix):
    aa={"A":0,"R":1,"N":2,"D":3,"C":4,"Q":5,"E":6,"G":7,"H":8,"I":9,"L":10,"K":11,"M":12,"F":13,"P":14,"S":15,"T":16,"W":17,"Y":18,"V":19}
    encoded_data = []
    pep_length = [8,9,10,11,12,13]
    for pair_i in binding_pairs:
        allele = pair_i[0][4:] #remove the "HLA-" prefix
        pep = pair_i[1] #Sequence of the peptide in the form of a string, like "AAVFPPLEP"
        score = pair_i[2] # transformed binding affinity values
        ba = pair_i[3] # transformed binding affinity values
        if allele in allele_pseudo_seqs.keys():
            if set(list(pep)).difference(list('ACDEFGHIKLMNPQRSTVWY')):
                print('Illegal peptides: AA error')
                continue
            if len(pep) not in pep_length:
                print('Illegal peptides: Length error')
                continue

            pep_blosum = []#Encoded peptide seuqence
            for residue_index in range(13):
                #Encode the peptide sequence in the 1-12 columns, with the N-terminal aligned to the left end
                #If the peptide is shorter than 12 residues, the remaining positions on
                #the rightare filled will zero-padding
                if residue_index < len(pep):
                    pep_blosum.append(blosum_matrix[aa[pep[residue_index]]])
                else:
                    pep_blosum.append(np.zeros(20))
            for residue_index in range(13):
                #Encode the peptide sequence in the 13-24 columns, with the C-terminal aligned to the right end
                #If the peptide is shorter than 12 residues, the remaining positions on
                #the left are filled will zero-padding
                if 13 - residue_index > len(pep):
                    pep_blosum.append(np.zeros(20))
                else:
                    pep_blosum.append(blosum_matrix[aa[pep[len(pep) - 13 + residue_index]]])
            pep_blosum_array = np.array(pep_blosum)

            ####
            allele_blosum = []  # Encoded allele seuqence
            allele_seq = allele_pseudo_seqs[allele]
            for aa_i in allele_seq:
                allele_blosum.append(blosum_matrix[aa[aa_i]])
            allele_blosum_array = np.array(allele_blosum)

            new_data = [pep_blosum_array, allele_blosum_array, score, allele, allele_seq, pep,ba]

            encoded_data.append(new_data)
    return encoded_data

encoded_data = hla_peptide_pairs_encode(binding_pairs,pseudo_seq_dict,blosum_matrix)
with open('data/encoded_allele_peptide.pkl', 'wb') as handle:
    pickle.dump(encoded_data, handle, protocol=pickle.HIGHEST_PROTOCOL)



