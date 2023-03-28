"""
Random pick negative dataset
"""

import re
import random
import numpy as np
import pandas as pd
from collections import Counter

def random_pick(seq, counts):
    x = random.randint(1, sum(counts))
    cumulative_probability = 0
    for item, item_probability in zip(seq, counts):
        cumulative_probability += item_probability
        if x <= cumulative_probability:
            break
    return item


def savetxtR(filename, data):
    import numpy as np
    dim = int(data.size / len(data))
    np.savetxt(filename, data, fmt=['%s'] * dim, delimiter='\t', newline='\n')


def read_proteome():
    '''
    Read the sequences of the proteins in the human proteome.
    Sequence data are stored in a fasta file
    Args:
        1. path: The input file containing sequence data of the proteome
            downloaded from the ensemble biomart FTP:
            ftp://ftp.ensembl.org/pub/release-90/fasta/homo_sapiens/.
    Return values:
        1. proteome: A dictionary whose keys are protein ensembl IDs
                    and values are protein sequences
    '''
    path = "data/original_data/Homo_sapiens.GRCh38.pep.all.fa"
    f = open(path, "r")
    proteome = {}
    seq = ""
    for tmp in f:
        if tmp[0] == ">":
            # Start reading the sequence for a new protein
            id = re.search("transcript:(ENST\d+)", tmp)
            seq = ""
        else:
            # Update the sequence. Use len(tmp)-1 because the last char in each line is \n
            seq += tmp[0:len(tmp) - 1]
            # update the sequence in the dictionary
            proteome[id.groups()[0]] = seq
    f.close()
    return proteome

def random_peptides():
    '''
    Randomly sample peptides from the proteome
    Args:
        1. proteome: A dictionary of the human proteome.
        Output of the function read_proteome
    Return values:
        1. peptides: Sampled peptides.
    '''
    proteome = read_proteome()
    # randomly generate peptides from the proteome
    iedb_csv = "data/mhc_ligand_pos.csv"
    iedb_df = pd.read_csv(iedb_csv, sep=',', skiprows=0, low_memory=False, dtype=object)
    iedb_df = np.array(iedb_df)

    all_positive_peptide = list(set([p[0] for p in iedb_df]))

    data_dict = {}
    for i in range(len(iedb_df)):
        allele = iedb_df[i][1]
        if allele not in data_dict.keys():
            data_dict[allele] = [iedb_df[i].tolist()]
        else:
            data_dict[allele].append(iedb_df[i].tolist())

    all_neg = []
    for allele in data_dict.keys():
        print(allele)
        # allele = 'HLA-B*41:06'
        pos_data = data_dict[allele]
        all_length = [len(pos_data[j][0]) for j in range(len(pos_data))]
        all_length_times = Counter(all_length)

        all_probabilities = []
        for kmer in [8, 9, 10, 11, 12, 13]:
            try:
                probabilities = all_length_times[kmer]
            except:
                probabilities = 0

            all_probabilities.append(probabilities)

        pep_seq = []
        k = 1 # Pos/Neg ratio
        while len(pep_seq) < k * len(pos_data):
            length = random_pick([8, 9, 10, 11, 12, 13], all_probabilities)
            accession = random.choice(list(proteome.keys()))
            protein = proteome[accession]
            # protein = random.choice(list(proteome.values()))
            if len(protein) < length:
                continue
            pep_start = random.randint(0, len(protein) - length)
            pep = protein[pep_start:pep_start + length]

            if set(list(pep)).difference(list('ACDEFGHIKLMNPQRSTVWY')):
                print('non offical peptide')
                continue
            if pep in all_positive_peptide:
                print('In positive peptide')
                continue
            if pep not in pep_seq:
                pep_seq.append([accession, pep])
                # pep_seq.append(pep)

        for pep_i in pep_seq:
            all_neg.append([allele,pep_i[1]])
    return all_neg

if __name__ == '__main__':
    all_neg = random_peptides()
    savetxtR("data/mhc_ligand_neg.txt",np.array(all_neg))
