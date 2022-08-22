'''
Author: QHGG
Date: 2021-08-02 22:46:25
LastEditTime: 2022-08-22 15:58:57
LastEditors: QHGG
Description: 
FilePath: /AlphaDrug/utils/docking.py
'''
import os
import subprocess
import re
from loguru import logger
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio import PDB
import json
import time
import numpy as np
from rdkit.ML.Descriptors import MoleculeDescriptors

def ProteinParser(pdbid):
    
    # others = os.listdir('./data/v2020-other-PL/')
    # orign = 'v2020-other-PL'
    # if pdbid not in others:
    #     orign = 'refined-set'
    orign = 'test_pdbs'
    parser = PDB.PDBParser(PERMISSIVE=1)
    structure = parser.get_structure(pdbid, './data/%s/%s/%s_protein.pdb'%(orign,pdbid,pdbid))
    ppb = PDB.PPBuilder()
    
    seq = ''
    for pp in ppb.build_peptides(structure):
        seq += pp.get_sequence()
    
    return seq

def CaculateAffinity(smi, file_protein='./1zys.pdb', file_lig_ref = './1zys_D_199.sdf', out_path = './', prefix=''):
    try:
        mol = Chem.MolFromSmiles(smi)
        m2=Chem.AddHs(mol)
        AllChem.EmbedMolecule(m2)
        m3 = Chem.RemoveHs(m2)
        file_output = os.path.join(out_path, prefix + str(time.time())+ '.pdb')
        Chem.MolToPDBFile(m3, file_output)

        # mol = Chem.MolFromPDBFile("test.pdb")
        # smile = Chem.MolToSmiles(mol)
        # logger.info(smile)
        # logger.info(smi)
        
        # file_drug="sdf_ligand_"+str(pdb_id)+str(i)+".sdf"
        smina_cmd_output = os.path.join(out_path, prefix + str(time.time()))
        launch_args = ["smina", "-r", file_protein, "-l", file_output, "--autobox_ligand", 
                    file_lig_ref, "--autobox_add", "10", "--seed", "1000", "--exhaustiveness", "9",">>", smina_cmd_output]
        # launch_args = ["smina", "-r", file_protein, "-l", file_output, "--autobox_ligand", 
        #             file_lig_ref, "--autobox_add", "10", "--seed", "1000", "--exhaustiveness", "9","-o", prefix+'dockres.pdb']
        # -o 1OYT-redock.pdbqt
        launch_string = ' '.join(launch_args)
        logger.info(launch_string)
        p = subprocess.Popen(launch_string, shell=True, stdout=subprocess.PIPE)
        p.communicate()

        affinity = 500
        with open(smina_cmd_output, 'r') as f:
            for lines in f.readlines():
                lines = lines.split()
                if len(lines) == 4 and lines[0] == '1':
                    affinity = float(lines[1])
                
        p = subprocess.Popen('rm -rf ' + smina_cmd_output, shell=True, stdout=subprocess.PIPE)
        p.communicate()
        p = subprocess.Popen('rm -rf ' + file_output, shell=True, stdout=subprocess.PIPE)
        p.communicate()
    
    except:
        affinity = 500

    if affinity == 500:
        logger.error('affinity error')

    
    return affinity

    