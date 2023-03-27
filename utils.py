import re

def read_fasta(fasta_file):
    try:
        fp = open(fasta_file)
    except IOError:
        exit()
    else:
        fp = open(fasta_file)
        lines = fp.readlines()
        fasta_dict = {}
        gene_id = ""
        for line in lines:
            if line[0] == '>':
                if gene_id != "":
                    fasta_dict[gene_id[1:]] = seq
                seq = ""
                gene_id = line.strip()
            else:
                seq += line.strip()
        fasta_dict[gene_id] = seq
    return fasta_dict

def read_blosum(path):
    '''
    Read the blosum matrix from the file blosum50.txt
    Args:
        1. path: path to the file blosum50.txt
    Return values:
        1. The blosum50 matrix
    '''
    f = open(path,"r")
    blosum = []
    for line in f:
        blosum.append([(float(i))/10 for i in re.split("\t",line)])
        #The values are rescaled by a factor of 1/10 to facilitate training
    f.close()
    return blosum



