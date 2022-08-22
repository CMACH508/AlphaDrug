'''
Author: QHGG
Date: 2022-08-22 15:54:41
LastEditTime: 2022-08-22 16:51:51
LastEditors: QHGG
Description: 
FilePath: /AlphaDrug/train.py
'''
import torch
import json
import os
import time
import argparse
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from loguru import logger
from utils.baseline import prepareDataset
from utils.baseline import loadConfig
from utils.log import prepareFolder, trainingVis
from model.Lmser_Transformerr import MFT as DrugTransformer
# from model.Transformer import MFT as DrugTransformer
# from model.Transformer_Encoder import MFT as DrugTransformer

def train(model, trainLoader, smiVoc, proVoc, device):
    batch = len(trainLoader)
    totalLoss = 0.0
    totalAcc = 0.0
    for protein, smile, label, proMask, smiMask in tqdm(trainLoader):
        protein = torch.as_tensor(protein).to(device)
        smile = torch.as_tensor(smile).to(device)
        proMask = torch.as_tensor(proMask).to(device)
        smiMask = torch.as_tensor(smiMask).to(device)
        label = torch.as_tensor(label).to(device)
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(model,smile.shape[1]).tolist()
        tgt_mask = [tgt_mask] * len(device_ids)
        tgt_mask = torch.as_tensor(tgt_mask).to(device)
        
        out = model(protein, smile, smiMask, proMask, tgt_mask)
        # tgt = torch.argmax(out, dim=-1)
        cacc = ((torch.eq(torch.argmax(out, dim=-1) * smiMask, label * smiMask).sum() - (smiMask.shape[0] * smiMask.shape[1] - (smiMask).sum())) / (smiMask).sum().float()).item()
        
        totalAcc += cacc
        loss = F.nll_loss(out.permute(0, 2, 1), label, ignore_index=smiVoc.index('^')) # mask padding

        # loss = vLoss
        totalLoss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # scheduler.step()

    avgLoss = round(totalLoss / batch, 3)
    avgAcc = round(totalAcc / batch, 3)

    return [avgAcc, avgLoss]


@torch.no_grad()
def valid(model, validLoader, smiVoc, proVoc, device):
    model.eval()
    
    batch = len(validLoader)
    totalLoss = 0
    totalAcc = 0
    for protein, smile, label, proMask, smiMask in tqdm(validLoader):
        protein = torch.as_tensor(protein).to(device)
        smile = torch.as_tensor(smile).to(device)
        proMask = torch.as_tensor(proMask).to(device)
        smiMask = torch.as_tensor(smiMask).to(device)
        label = torch.as_tensor(label).to(device)
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(model,smile.shape[1]).tolist()
        tgt_mask = [tgt_mask] * len(device_ids)
        tgt_mask = torch.as_tensor(tgt_mask).to(device)

        out = model(protein, smile, smiMask, proMask, tgt_mask)
        
        # totalAcc += ((torch.eq(torch.argmax(out, dim=2) * smiMask, label * smiMask).sum() - (smiMask.shape[0] * smiMask.shape[1] - smiMask.sum())) / smiMask.sum()).item()
        
        cacc = ((torch.eq(torch.argmax(out, dim=-1) * smiMask, label * smiMask).sum() - (smiMask.shape[0] * smiMask.shape[1] - (smiMask).sum())) / (smiMask).sum().float()).item()
        
        totalAcc += cacc

        loss = F.nll_loss(out.permute(0, 2, 1), label, ignore_index=smiVoc.index('^')) # mask padding

        # loss = vLoss
        totalLoss += loss.item()
        
    avgLoss = round(totalLoss / batch, 3)
    avgAcc = round(totalAcc / batch, 3)

    return [avgAcc, avgLoss]
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='settings')
    
    parser.add_argument('--layers', type=int, default=4, help='transformer layers')
    parser.add_argument('-l', action="store_true", help='learning rate')
    parser.add_argument('--epoch', type=int, default=501, help='epochs')
    parser.add_argument('--device', type=str, default='0,1,2,3', help='device')
    parser.add_argument('--pretrain', type=str, default='', help='pretrain model path')
    parser.add_argument('--bs', type=int, default=32, help='bs')
    parser.add_argument('--note', type=str, default='', help='note')
    args = parser.parse_args()

    startTime = time.time()

    config = loadConfig(args)
    
    exp_folder, model_folder, vis_folder = prepareFolder()
    logger.success(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_ids = [int(i) for i in args.device.split(',') if i!=''] # 10卡机
    batchSize = args.bs * len(device_ids)
    epoch = args.epoch
    lr = 1e-3

    config.batchSize = batchSize

    trainLoader, validLoader = prepareDataset(config)
    settings = {
            'remark': args.note,
            'smiVoc': config.smiVoc,
            'proVoc': config.proVoc,
            'smiMaxLen': config.smiMaxLen,
            'proMaxLen': config.proMaxLen,
            'smiPaddingIdx': config.smiVoc.index('^'),
            'proPaddingIdx': config.proVoc.index('^'),
            'smi_voc_len': len(config.smiVoc),
            'pro_voc_len': len(config.proVoc),
            'batchSize': config.batchSize,
            'epoch': epoch,
            'lr': lr,
            'd_model': 96,
            'dim_feedforward': 256,
            'num_layers': args.layers, 
            'nhead': 4,
        }
    logger.info(settings)
    # 写入本次训练配置
    with open((exp_folder + 'settings.json'), 'w') as f:
        json.dump(settings, f)

    model = DrugTransformer(**settings)
    model = torch.nn.DataParallel(model, device_ids=device_ids) # 指定要用到的设备
    model = model.to(device) # 模型加载到设备0

    if len(args.pretrain):
        model.load_state_dict(torch.load(args.pretrain))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.99, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,5,eta_min=0,last_epoch=-1)
    
    # optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
    # scheduler = nn.DataParallel(scheduler, device_ids=device_ids)


    propertys = ['accuracy', 'loss', ]
    prefixs = ['training', 'validation']
    columns = [' '.join([pre, pro]) for pre in prefixs for pro in propertys]
    logdf = pd.DataFrame({}, columns=columns)


    for i in range(epoch):
        logger.info('EPOCH: {} 训练'.format(i))
        d1 = train(model, trainLoader, config.smiVoc, config.proVoc, device)
        
        logger.info('EPOCH: {} 验证'.format(i))
        d2  = valid(model, validLoader, config.smiVoc, config.proVoc, device)
        
        logdf = logdf.append(pd.DataFrame([d1+d2], columns=columns), ignore_index=True)
        trainingVis(logdf, batchSize, lr, vis_folder)
        
        if args.l:
            scheduler.step()

        # if i % 10 == 0:
        torch.save(model.state_dict(), model_folder + '{}.pt'.format(i))
        # for name,parameters in model.named_parameters():
        #     print(name,':',parameters.size(), parameters)
        logdf.to_csv(exp_folder+'logs/logdata')

    endTime = time.time()
    logger.info('time: {} h'.format((endTime - startTime) / 3600))