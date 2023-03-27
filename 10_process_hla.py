"""
Download data and process data
HLA allele sequence data
    2.1 sequence data are from IPD-IMGT/HLA: https://github.com/ANHIG/IMGTHLA/
"""
import json
from utils import *

def allele_seq(path):
    '''
    Read the sequece of each allele. If there are multiple subtypes, choose the first one.
    Args:
        1. path: path to the data file hla_prot_fasta
    Return values:
        1. A dictionary whose keys are the name of MHC alleles and the corresponding dict
        values are amino acid sequences of those alleles.
    '''
    seq_dict = {}
    f = open(path, "r")
    allele = None
    for line in f:
        if line[0] == ">":  # A new allele
            match = re.search("(\w\*\d+:\d+)", line)  # The standard allele id are like "A*01:01:..."

            # For alleles with the same two digits, like A*01:01:01 and A*01:01:02, we take the first one as the representative
            allele = None  # While reading the sequence of the same alleles from different
            # lines of the file, allele is not None, so that the sequences of each line
            # will be added to the end of the correspondong sequence
            # Some times we run into alleles with incorrect names, so that allele is set to None
            # and match == None so allele will not be reset, then the following lines of sequences
            # will not be recorded
            if match != None:  # If the current allele has a name with the correct format
                if match.groups()[0] not in seq_dict.keys():
                    allele = match.groups()[0]  # A new allele
                    seq_dict[allele] = ""  # And its sequence
        elif allele != None:
            seq_dict[allele] = seq_dict[allele] + line[:-1]
            # Each line contains only 60 redidues, so add the sequence of the current line
            # to the end of the corresponding sequence

    for allele in list(seq_dict.keys()):
        if len(seq_dict[allele]) < len(seq_dict['B*07:02']):
            # Some sequences lack certain parts like the leader peptide, and cannot
            # be aligned to other sequences well. The ones longer than B*07:02 can
            # be aligned well with the majority of HLA A and B alleles, (see this in uniprot)
            seq_dict.pop(allele)

    return seq_dict

##
def pseudo_seq(seq_dict):
    '''
    Generate the pseudo sequence for each allele
    Args:
        1. seq_dict: the output of allele_seq()
    Return values:
        1. A dictionary whose keys are the name of MHC alleles and the corresponding dict
        values are the pseudo-sequences of those alleles.
    '''

    pseq_dict = {}  # pseudo sequence dictionary

    # First exclude the alleles that cannot align to the majority of the alleles
    # Most of the alleles can be directly aligned to each other, with a few exceptions having particularly short
    # Sequences and low homogeneity with other alleles

    # Remove the sequence of the signal peptide
    for allele in seq_dict.keys():
        seq_dict[allele] = seq_dict[allele][24:]

    # Actual indices of selected residues
    residue_indices = [7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81, 84, 95, 97, 99, 114, 116, 118,
                       143, 147, 150,152, 156, 158, 159, 163, 167, 171]

    # Indices of these residues in a python list
    residue_indices = [i - 1 for i in residue_indices]

    # Now encode the MHC sequences into pseudo-sequences.
    for allele in seq_dict.keys():
        new_pseq = []
        pseq = ""
        for index in residue_indices:
            pseq += seq_dict[allele][index]
        pseq_dict[allele] = pseq

    return pseq_dict

### allele seq data, fasta format
allele_seq_dict = allele_seq("data/original_data/hla_prot.fasta")
pseudo_seq_dict = pseudo_seq(allele_seq_dict)
with open("data/allele_pseudo_seq.txt", "w") as fp:
    json.dump(pseudo_seq_dict, fp)  # encode dict into JSON
print("Done writing dict into .json file")



