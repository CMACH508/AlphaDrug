'''
Author: QHGG
Date: 2021-08-02 22:24:17
LastEditTime: 2022-08-22 16:49:40
LastEditors: QHGG
Description: 
FilePath: /AlphaDrug/beamsearch.py
'''
import numpy as np
import torch
import json
import shutil
import time
import os
import argparse
from torch import nn
from utils.log import timeLable
from model.Lmser_Transformerr import MFT as DrugTransformer
# from model.Transformer import MFT as DrugTransformer
# from model.Transformer_Encoder import MFT as DrugTransformer

from rdkit import Chem
from loguru import logger
from utils.docking import CaculateAffinity, ProteinParser

class Node:

    def __init__(self, path=[], wins=0):
        self.wins = wins
        self.path = path  #MCTS 路径
      
def JudgePath(path, smiMaxLen):
    return path[-1] != '$' and len(path) < smiMaxLen

def check_node(node, k):
    affinity = 500
    if node.path[-1] != '$':
        return affinity, ''.join(node.path[1:])

    smile = ''.join(node.path[1:-1])
    try:
        m = Chem.MolFromSmiles(smile)
    except:
        pass
    if m:
        logger.info(smile)
        affinity = CaculateAffinity(smile, file_protein=pro_file[k], file_lig_ref=ligand_file[k], out_path=resFolderPath)
        
    return affinity, smile

@torch.no_grad()
def sample(model, path, vocabulary, proVoc, smiMaxLen, proMaxLen, device, sampleTimes, protein_seq):
    model.eval()

    pathList = path[:]
    length = len(pathList)
    pathList.extend(['^'] * (smiMaxLen - length))

    protein = '&' + protein_seq +'$'
    proList = list(protein)
    lp = len(protein)

    proList.extend(['^'] * (proMaxLen - lp))
    
    proteinInput = [proVoc.index(pro) for pro in proList]
    currentInput = [vocabulary.index(smi) for smi in pathList]
    
    src = torch.as_tensor([proteinInput]).to(device)
    tgt = torch.as_tensor([currentInput]).to(device)

    smiMask = [1] * length + [0] * (smiMaxLen - length)
    smiMask = torch.as_tensor([smiMask]).to(device)
    proMask = [1] * lp + [0] * (proMaxLen - lp)
    proMask = torch.as_tensor([proMask]).to(device)

    tgt_mask = nn.Transformer.generate_square_subsequent_mask(model, smiMaxLen).tolist()
    tgt_mask = [tgt_mask] * 1
    tgt_mask = torch.as_tensor(tgt_mask).to(device)

    sl = length - 1
    out = model(src, tgt, smiMask, proMask, tgt_mask)[:, sl, :]
    out = out.tolist()[0]
    pr = np.exp(out) / np.sum(np.exp(out))
    prList = np.random.multinomial(1, pr, sampleTimes)
    
    indices = list(set(np.argmax(prList, axis=1)))
    
    atomList = [vocabulary[i] for i in indices]
    logpList = [np.log(pr[i] + 1e-10) for i in indices]
    
    atomListExpanded = []
    logpListExpanded = []
    for idx, atom in enumerate(atomList):
        if atom == '&' or atom == '^':
            continue
        atomListExpanded.append(atom)
        logpListExpanded.append(logpList[idx])
    # logger.info(atomListExpanded)
    return atomListExpanded, logpListExpanded

# @logger.catch
def BeamSearch(experimentId, modelName, root, k, beamSize=10):
    device = torch.device("cuda:"+args.device if torch.cuda.is_available() else "cpu")

    device_ids = [int(args.device)] # 10卡机

    with open(os.path.join(experimentId, 'settings.json'), 'r') as f:
        s = json.load(f)
        
    model = DrugTransformer(**s)
    model = torch.nn.DataParallel(model, device_ids=device_ids) # 指定要用到的设备
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(experimentId +'/model/'+ modelName, map_location={'cuda:0':'cuda:'+args.device}))
    else:
        model.load_state_dict(torch.load(experimentId +'/model/'+ modelName, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    
    vocabulary = s['smiVoc']
    proVoc = s['proVoc']
    smiMaxLen = int(s['smiMaxLen'])
    proMaxLen = int(s['proMaxLen'])

    groundIndex = 0 # MCTS Node唯一计数
    rootNode = Node(path=['&'])
    allScore = []
    allValidSmiles = []
    allSmiles = []

    successNode = []
    nodeList = [rootNode]
    while(len(successNode) < beamSize and len(nodeList)):
        beam_expand_node = []

        for node in nodeList:
            if JudgePath(node.path, smiMaxLen):
                atomListExpanded, logpListExpanded = sample(model, node.path, vocabulary, proVoc, smiMaxLen, proMaxLen, device,30, protein_seq)
                beam_expand_node.extend([Node(path=node.path + [atom], wins=node.wins+logpListExpanded[idx]) for idx, atom in enumerate(atomListExpanded)])
            else:
                affinity, smile = check_node(node, k)
                allSmiles.append(smile)
                if affinity != 500:
                    allValidSmiles.append(smile)
                    allScore.append(-affinity)
                    successNode.append(node)
                    groundIndex += 1
                    logger.info(str(groundIndex) + '\t' + ''.join(node.path) + '\t' + str(-affinity))
        topk = min(len(beam_expand_node), beamSize - len(successNode))
        beam_expand_node.sort(key=lambda node:node.wins, reverse=True)
        nodeList = beam_expand_node[:topk]
        


    
    # 写入结果
    with open(os.path.join(root, test_pdblist[k]+ '_' + modelName + '-bsres.json'), 'w') as f:
        json.dump({
            'pdbid': test_pdblist[k],
            'score': allScore,
            'validSmiles': allValidSmiles,
            'allSmiles': allSmiles
        }, f)
        
    logger.success('valid: {}'.format(len(allValidSmiles) / len(allSmiles)))
    
    return allScore
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=0)
    parser.add_argument('--device', type=str, default='3')
    parser.add_argument('-bs', type=int, default=10)
    parser.add_argument('--source', type=str, default='new')
    parser.add_argument('-p', type=str, default='LT', help='pretrained model')

    args = parser.parse_args()

    if args.source == 'new':
        test_pdblist = sorted(os.listdir('./data/test_pdbs/'))
        pro_file = ['./data/test_pdbs/%s/%s_protein.pdb'%(pdb,pdb) for pdb in test_pdblist]
        ligand_file = ['./data/test_pdbs/%s/%s_ligand.sdf'%(pdb,pdb) for pdb in test_pdblist]
        protein_seq = ProteinParser(test_pdblist[args.k])
    else:
        raise NotImplementedError('Unknown source: %s' % args.source)
    
    
    beamSize = args.bs
    experimentId = os.path.join('experiment', args.p)
    ST = time.time()
    m = '30.pt'
    mPath = 'bs_%s_%s_%s_%s_%s'%(beamSize, timeLable(), m, args.k, test_pdblist[args.k])
    resFolderPath = os.path.join(experimentId, mPath)
    
    if not os.path.isdir(resFolderPath):
        os.mkdir(resFolderPath)
    logger.add(resFolderPath+"/{time}.log")
    logger.info('k='+str(args.k))
    
    shutil.copyfile('./beamsearch.py',resFolderPath + '/beamsearch.py')
    
    if len(protein_seq) > 999:
        logger.info('skipping %s'%test_pdblist[args.k])
    else:
        score = BeamSearch(experimentId, m, resFolderPath, args.k, beamSize=beamSize)
            
    ET = time.time()
    logger.info('time {}'.format((ET-ST)//60))
