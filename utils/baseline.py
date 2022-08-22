'''
Author: QHGG
Date: 2021-11-03 22:40:24
LastEditTime: 2022-08-22 17:02:07
LastEditors: QHGG
Description: dataloader with coords
FilePath: /AlphaDrug/utils/baseline.py
'''

import json
import re
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from scipy import stats
from easydict import EasyDict

class MyDataset(Dataset):
    
    def __init__(self, data):
        proIndices, smiIndices, labelIndices, proMask, smiMask = data
        self._len = len(proIndices)
        self.x = proIndices
        self.y = smiIndices
        self.label = labelIndices
        self.proMask = proMask
        self.smiMask = smiMask
    
    def __getitem__(self, idx):
        proMask = [1.0] * self.proMask[idx] + [0.0] * (len(self.x[idx]) - self.proMask[idx])
        smiMask = [1.0] * self.smiMask[idx] + [0.0] * (len(self.label[idx]) - self.smiMask[idx])
        
        return self.x[idx], self.y[idx], self.label[idx], np.array(proMask).astype(int), \
        np.array(smiMask).astype(int)

    def __len__(self):
        return self._len

def prepareDataset(config):
    train = prepareData(config, 'train')
    valid = prepareData(config, 'valid')
    trainLoader = DataLoader(MyDataset(train), shuffle=True, batch_size=config.batchSize, drop_last=False)
    validLoader = DataLoader(MyDataset(valid), shuffle=False, batch_size=config.batchSize, drop_last=False)

    return trainLoader, validLoader

def padPocCoords(Coords, MaxLen):
    return [[0.0, 0.0, 0.0]] + Coords + (MaxLen - 1 - len(Coords)) * [[0.0,0.0,0.0]]

def padLabelPocCoords(Coords, MaxLen):
    return Coords + (MaxLen - len(Coords)) * [[0.0,0.0,0.0]]

def smilesCoordsMask(mask, MaxLen):
    return mask + (MaxLen - len(mask)) * [0]

def readBindingDB(PATH):
    i = 0
    n = 0
    pdbidArr = []
    pocSeqArr = []
    smiArr = []
    affinityArr = []
    with open(PATH, 'r') as f:
        for lines in tqdm(f.readlines()):
            n+=1
            arr = lines.split(' ')
            pocSeq = arr[0] 
            print(lines.split('\t'))
            smi = arr[1][:-1]
            try:
                mol = Chem.MolFromSmiles(smi)
                mol = Chem.RemoveHs(mol, sanitize=False)
                smi = Chem.MolToSmiles(mol)
                if '%' in smi:
                    continue
                if '.' in smi:
                    continue
            except:
                continue

            pdbidArr.append('xxxx')
            pocSeqArr.append(pocSeq)
            smiArr.append(smi)
            affinityArr.append(0.0)

            i+=1


    print(i, n, i / n)

    data = pd.DataFrame({
        'pdbid': pdbidArr,
        'protein': pocSeqArr,
        'smile': smiArr,
        'affinity': affinityArr,
    })
    return data

def prepareData(config, orign):
    logger.info('prepare %s data' % orign)
    with open('./data/train-val-split.json', 'r') as f:
        data_config = json.load(f)
    slices = data_config[orign]
    data = pd.read_csv('./data/train-val-data.tsv', sep = '\t')

    # 小样本测试
    slices = [i for i in slices if i < 1000]
    
    data = data.loc[slices]
    smiArr = data['smiles'].apply(splitSmi).tolist()
    proArr = data['protein'].apply(list).tolist()
    
    logger.info('prepare %s smiles' % orign)
    smiIndices, labelIndices, smiMask = fetchIndices(smiArr, config.smiVoc, config.smiMaxLen)
    
    logger.info('prepare %s proteins' % orign)
    proIndices, _, proMask = fetchIndices(proArr, config.proVoc, config.proMaxLen)
    
    return proIndices, smiIndices, labelIndices, proMask, smiMask

def loadConfig(args):
    logger.info('prepare data config...')
    data = pd.read_csv('./data/train-val-data.tsv', sep = '\t')
    
    proMaxLen = max(list(data['protein'].apply(len))) + 2
    smiMaxLen = max(list(data['smiles'].apply(splitSmi).apply(len))) + 2

    pros_split = data['protein'].apply(list)
    proVoc = sorted(list(set([i for j in pros_split for i in j])) + ['&', '$', '^'])

    smiles_split = data['smiles'].apply(splitSmi)
    smiVoc = sorted(list(set([i for j in smiles_split for i in j])) + ['&', '$', '^'])

    return EasyDict({
        'proMaxLen': proMaxLen,
        'smiMaxLen': smiMaxLen,
        'proVoc': proVoc,
        'smiVoc': smiVoc,
        'args': args
    })

def pearsonr_ci(x,y,alpha=0.05):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''
    x = np.array(x)
    y = np.array(y)
    r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi
    
def splitSmi(smi):
    '''
    description: 将smiles拆解为最小单元
    param {*} smi
    return {*}
    '''
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return tokens


def fetchIndices(smiArr, smiVoc, smiMaxLen):
    smiIndices = []
    labelIndices = []
    mask = []
    # padding symbol: ^ ; end symbol: $ ; start symbol: &
    for smi in tqdm(smiArr):
        smiSplit = smi[:]
        smiSplit.insert(0, '&')
        smiSplit.append('$')

        labelSmi = smiSplit[1:]
        mask.append(len(smiSplit))

        smiSplit.extend(['^'] * (smiMaxLen - len(smiSplit)))
        smiIndices.append([smiVoc.index(smi) for smi in smiSplit])

        labelSmi.extend(['^'] * (smiMaxLen - len(labelSmi)))
        labelIndices.append([smiVoc.index(smi) for smi in labelSmi])

    return np.array(smiIndices), np.array(labelIndices), np.array(mask)



    
if __name__ == '__main__':
    pass

    
