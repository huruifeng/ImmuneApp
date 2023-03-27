"""
Download data and process data
MHC class I binding affinity data:
    1.1 IEDB13 website at http://tools.immuneepitope.org/mhci/download (http://tools.immuneepitope.org/static/main/binding_data_2013.zip)
        Ref1: Kim Y. et al.(2014) Dataset size and composition impact the reliability of performance benchmarks for peptide–MHC binding predictions. BMC Bioinformatics, 15, 241.
        Ref2: Vita R. et al.(2015) The immune epitope database (iedb) 3.0. Nucleic Acids Res., 43, D405–D412.
    1.2 Binding data from the following publication()
        Ref: Pearson H. et al.  (2016) Mhc class i-associated peptides derive from selective regions of the human genome. J. Clin. Investig., 126, 4690–4701.
"""
import numpy as np
import pandas as pd

## Binding data
binding_data1 = pd.read_csv("data/original_data/bdata.20130222.mhci.txt",index_col=None, header=0, sep="\t")
binding_data1 = binding_data1.loc[binding_data1.species=="human",["mhc","sequence","peptide_length","meas"]]
binding_data1.rename(columns={"mhc": "allele", "sequence": "peptide","peptide_length": "peplen","meas": "affinity"},inplace=True)
binding_data1.dropna(1,inplace=True)
binding_data1.loc[:,"dataset"] = "iedb13"

binding_data2 = pd.read_csv("data/original_data/Pearson.csv",index_col=None, skiprows=[0], header=0, sep=",")
binding_data2 = binding_data2.loc[:,["Allele","Peptide Sequence","Length","Binding Affinity"]]
binding_data2.rename(columns={"Allele": "allele", "Peptide Sequence": "peptide","Length": "peplen","Binding Affinity": "affinity"},inplace=True)
binding_data2.dropna(0,inplace=True)
binding_data2.loc[:,"dataset"] = "external"

allele_ls = [i[:-3]+":"+i[-2:] for i in binding_data2.allele.tolist()]
binding_data2.loc[:,"allele"] = allele_ls
combined_df = pd.concat([binding_data1,binding_data2],axis=0)
combined_df.drop_duplicates(subset=["allele", "peptide", "peplen", "affinity"])
combined_df.loc[:,"score"] = 1-(np.log10(combined_df.affinity)/np.log10(50000))
combined_df.to_csv("data/binding_data.csv",index=False)

allele_ls = set(combined_df.allele.tolist())

####



