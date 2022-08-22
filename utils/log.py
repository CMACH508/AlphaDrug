'''
Author: QHGG
Date: 2021-07-31 19:07:28
LastEditTime: 2022-08-22 15:58:19
LastEditors: QHGG
Description: 
FilePath: /AlphaDrug/utils/log.py
'''

import time
import json
import os
import torch
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import socket
from graphviz import Digraph

from loguru import logger
from easydict import EasyDict
matplotlib.use("Agg")


def path(name):
    return os.path.dirname(__file__) + '/' + name
    
def timeLable():
    return  time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

def prepareFolder():
    root = 'experiment/'
    # exp_folder = 'testTransformer/'
    exp_folder = root + 'testTransformer' + timeLable() + '_' + socket.gethostname() +'/'
    # exp_folder = root + exp_folder
    mcts_folder = exp_folder + 'mcts/'
    logs_folder = exp_folder + 'logs/'
    model_folder = exp_folder + 'model/'
    vis_folder = exp_folder + 'vis/'
    if not os.path.isdir(root):
        os.mkdir(root)
    if not os.path.isdir(exp_folder):
        os.mkdir(exp_folder)
    if not os.path.isdir(logs_folder):
        os.mkdir(logs_folder)
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)
    if not os.path.isdir(vis_folder):
        os.mkdir(vis_folder)
    # shutil.copyfile('../exp/model/model0.json',model_folder + 'model0.json')
    # shutil.copyfile('../exp/model/model0.h5',model_folder + 'model0.h5')
    logger.add(logs_folder + "{time}.log")
    
    return exp_folder, model_folder, vis_folder

def trainingVis(df, lr, batch_size, path, time=time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())):
    # make a figure
    fig = plt.figure(figsize=(16,9))
    
    # subplot loss
    ax3 = fig.add_subplot(121)
    ax3.plot(df['training loss'].tolist(), label='train_loss')
    ax3.plot(df['validation loss'].tolist(), label='val_loss')
    # ax3.plot(df['testing loss'].tolist(), label='test_loss')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Loss')
    ax3.set_title('Total Loss')
    ax3.legend()

    # subplot acc
    ax4 = fig.add_subplot(122)
    ax4.plot(df['training accuracy'].tolist(),label='train_acc')
    ax4.plot(df['validation accuracy'].tolist(),label='val_acc')
    # ax4.plot(df['testing accuracy'].tolist(),label='test_acc')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Accuracy')
    ax4.legend()


    plt.tight_layout()
    plt.savefig(path + time + ' ' + str(lr) + ' ' + str(batch_size) + '.png')
    plt.close()

def readSettings(experimentId):
    with open(os.path.join(experimentId, 'settings.json'), 'r') as f:
        s = json.load(f)
    return EasyDict(s)

def GenLabel(root):
    return '{} {} {:.2f}'.format(root.index, root.path[-1], root.CaculatePUCT())

def GenInterLabel(root,QMAX, QMIN, QE):
    c = 1.5
    if QMAX == QMIN:
        logger.error('QMAX == QMIN')
        q = 0
    else:
        if root.visits:
            q = (root.wins/root.visits - QMIN) / (QMAX - QMIN)
        else:
            q = (QE - QMIN) / (QMAX - QMIN)

    if root.parentNode:
        u = c*root.p*np.sqrt(root.parentNode.visits)/(1+root.visits)
    else:
        u = 0
    return '{}*{}*{}*{}'.format(root.index, root.visits,np.round(q,2), np.round(u,2))

def VisualizeMCTS(rootnode, modelName, path, times):
    path = os.path.join(path, 'gv')
    if not os.path.isdir(path):
        os.mkdir(path)
    u = Digraph('unix', filename='unix.gv', node_attr={'color': 'lightblue2', 'style': 'filled'})
    u.attr(size='6,6')
    u.node(str(rootnode.index), label = GenLabel(rootnode))
    Draw(u,rootnode)
    u.save('%s_%s.gv'%(times, str(modelName)), path)

def VisualizeInterMCTS(rootnode, modelName, path, times, QMAX, QMIN, QE):
    path = os.path.join(path, 'gv')
    if not os.path.isdir(path):
        os.mkdir(path)
    u = Digraph('unix', filename='unix.gv', node_attr={'color': 'lightblue2', 'style': 'filled'})
    u.attr(size='6,6')
    u.node(str(rootnode.index), label = GenInterLabel(rootnode,QMAX, QMIN, QE))
    DrawInter(u,rootnode,QMAX, QMIN, QE)
    u.save('%s_%s.gv'%(times, str(modelName)), path)

def DrawInter(g, root,QMAX, QMIN, QE):
    for child in root.childNodes:
        g.node(str(child.index), label = GenInterLabel(child,QMAX, QMIN, QE))
        g.edge(str(root.index), str(child.index))
        DrawInter(g, child,QMAX, QMIN, QE)

def Draw(g, root):
    for child in root.childNodes:
        g.node(str(child.index), label = GenLabel(child))
        g.edge(str(root.index), str(child.index))
        Draw(g, child)

def saveMCTSRes(path, data):
    with open(os.path.join(path, 'res.json'), 'w') as f:
        json.dump(data, f)

    
if __name__ == '__main__':
    print(socket.gethostname())