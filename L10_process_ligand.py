"""
Download data and process data
The HLA-I presenting ligands were acquired from IEDB at https://www.iedb.org/database_export_v3.php (file name: mhc_ligand_full_single_file.zip
"""

import json
import pandas as pd

with open("data/allele_pseudo_seq.json", "r") as fp:
    pseudo_seq_dict = json.load(fp)  # encode dict into JSON
allele_ls = ["HLA-"+i for i in list(pseudo_seq_dict.keys())]

ligand_df = pd.read_csv("data/original_data/mhc_ligand_full.csv",index_col=None,header=[0,1])
## field Assay Group = “ligand presentation”, field Technique = “cellular MHC/mass spectrometry” or “mass spectrometry”, and “secreted MHC/mass spectrometry”).
condition = (ligand_df["Assay"]["Assay Group"]=="ligand presentation") & \
            (ligand_df["Epitope"]["Object Type"] == "Linear peptide") & \
            (ligand_df["Assay"]["Method/Technique"].isin(['cellular MHC/mass spectrometry','mass spectrometry','secreted MHC/mass spectrometry'])) & \
            (ligand_df["Host"]["Name"].isin(['Homo sapiens (human)','human (Homo sapiens)','Homo sapiens'])) & \
            (ligand_df["MHC"]["MHC allele class"] == "I")
ligand_df=ligand_df.loc[condition,:]
ligand_df_top100 = ligand_df.iloc[:100,:]
ligand_df=ligand_df.loc[:,[("Epitope","Description"),("MHC","Allele Name")]]
ligand_df.columns = ["Peptide","Allele"]
ligand_df_top100 = ligand_df.iloc[:100,:]

ligand_df.loc[:,"peplen"] = ligand_df.Peptide.str.len()
ligand_df = ligand_df.loc[(ligand_df.peplen>=8)&(ligand_df.peplen<=13),:]

ligand_df.drop(["peplen"], axis=1,inplace=True)

ligand_df = ligand_df.loc[ligand_df.Allele.isin(allele_ls),:]
ligand_df.to_csv("data/mhc_ligand_pos.csv",index=None)
