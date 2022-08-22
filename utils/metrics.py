'''
Author: QHGG
Date: 2022-03-31 22:07:08
LastEditTime: 2022-08-22 15:57:44
LastEditors: QHGG
Description: metrics
FilePath: /AlphaDrug/utils/metrics.py
'''


import os, sys, pickle, gzip, json
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
sys.path.append(os.path.join(RDConfig.RDContribDir, 'NP_Score'))
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, AllChem
import sascorer
import npscorer
from loguru import logger

# logger.info(Descriptors._descList)
# des_name = [name[0] for name in Descriptors._descList]
# logger.info(des_name)
# fscore = npscorer.readNPModel()
fscore = pickle.load(gzip.open(os.path.join(RDConfig.RDContribDir, 'NP_Score') + '/publicnp.model.gz'))





def calcScore(mol):
    """
        return MolLogP, qed, sa_score, np_score, docking_score
    """
    des_list = ['MolLogP', 'qed', 'TPSA', 'MolWt']
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des_list)
    
    MolLogP, qed, tpsa, MolWt = calculator.CalcDescriptors(mol)
    sa_score = sascorer.calculateScore(mol)
    np_score = npscorer.scoreMol(mol, fscore)
    # docking_score = CaculateAffinity(Chem.MolToSmiles(mol))
    # return MolLogP, qed, sa_score, np_score, docking_score
    return MolLogP, qed, tpsa, MolWt, sa_score, np_score, 

def visDensity(path_list, out_path):
    dataArr = []
    label = ['LogP', 'QED', 'SA', 'NP', 'MolWt', 'Docking']
    color = ['#DA4453', '#4A89DC', '#967ADC', '#D770AD', '#37BC9B']
    legend = ['ligann', 'ours']
    for data_path in path_list:
        with open(data_path, 'r') as f:
            s = json.load(f)
        data = []
        score = s['score']
        smiles = s['validSmiles']

        dict = {}
        no_repeat_smiles = []
        for i, smi in enumerate(smiles):
            if smi in dict:
                dict[smi].append(float(score[i]))
            else:
                dict[smi] = [float(score[i])]
                no_repeat_smiles.append(smi)
        
        for smi, score_arr in tqdm(dict.items()):
            mol = Chem.MolFromSmiles(smi)
            MolLogP, qed, tpsa, MolWt, sa_score, np_score = calcScore(mol)
            data.append([MolLogP, qed, sa_score, np_score, MolWt, np.mean(score_arr)]) 
            # data.append([MolLogP, qed, tpsa, sa_score, np_score]) 
        data = np.array(data)
        logger.info(np.mean(data, axis=0))
        logger.info(np.var(data, axis=0))
        indices = np.random.choice(a=len(data), size=len(data), replace=False, p=None)
        logger.info(calcTanimotoSimilarity([no_repeat_smiles]))
        dataArr.append(data[indices][:,:].T)
    dataArr = np.array(dataArr).transpose(1, 0, 2)
    # print(dataArr.shape)
    plt.figure(figsize=(20, 12))
    
    for i in range(dataArr.shape[0]):
        plt.subplot(231 + i)
        for j in range(dataArr.shape[1]):
            ax = sn.kdeplot(dataArr[i, j, :],color=color[j],shade=True)
        plt.xlabel(label[i], fontsize=18)
        plt.ylabel(' ')
        plt.xticks(size=14)
        plt.yticks(size=14)
        plt.legend(legend)
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
        
    # data = [calcScore(Chem.MolFromSmiles(smi))[0] for smi in s['validSmiles']]
    # res = sn.kdeplot(data,color='green',shade=True, x="total_bill")

def calcTanimotoSimilarityPairs(s1, s2):
    fp1 = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s1), 2, nBits=1024)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s2), 2, nBits=1024)
    return DataStructs.FingerprintSimilarity(fp1,fp2)



def calcTanimotoSimilarity(smiles_arr):
    fpsLlist = []
    for smiles in smiles_arr:
        fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), 2, nBits=1024) for smile in smiles]
        fpsLlist.append(fps)
    
    data = []
    for fps1 in tqdm(fpsLlist):
        for fps2 in fpsLlist:
            IntDiv = []
            for fp1 in fps1:
                for fp2 in fps2:
                    IntDiv.append(DataStructs.FingerprintSimilarity(fp1,fp2))
            data.append(1- np.sqrt(np.sum(IntDiv)/(len(fps1)*len(fps2))))
    
    return data

