"""
Download data and process data
The HLA-I presenting ligands were acquired from IEDB at https://www.iedb.org/database_export_v3.php (file name: mhc_ligand_full_single_file.zip
"""
import pandas as pd


ligand_df = pd.read_csv("data/original_data/mhc_ligand_full.csv",index_col=None,header=[0,1])
## field Assay Group = “ligand presentation”, field Technique = “cellular MHC/mass spectrometry” or “mass spectrometry”, and “secreted MHC/mass spectrometry”).
condition = (ligand_df["Assay"]["Assay Group"]=="ligand presentation") & \
            (ligand_df["Assay"]["Method/Technique"].isin(['cellular MHC/mass spectrometry','mass spectrometry','secreted MHC/mass spectrometry'])) & \
            (ligand_df["Host"]["Name"].isin(['Homo sapiens (human)','human (Homo sapiens)','Homo sapiens']))
ligand_df=ligand_df.loc[condition,:]
