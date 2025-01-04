import sys
sys.path.append('../')
import dgl
import dgl.function as fn
import os
os.environ["NUMEXPR_MAX_THREADS"] = "64" 
import multiprocessing as mp
from tqdm import tqdm
import pdb
import random
import numpy as np
import torch
import torch.nn as nn
import logging
logging.basicConfig(stream = sys.stdout, level = logging.INFO)
from utils.parser import parse_args
from models.model import Proposed_model
from models.Predictor import Predictor
import pickle
from tqdm import tqdm


game_cold = set(torch.load("./data_exist/game_cold_2.pth")[0].tolist())



def ConventionalCoverage(ls_tensor, n_user, k):
    

    set1 = set()
    set2 = set()
    ls_tail = []


    for i in tqdm(range(n_user)):
        ls = ls_tensor[i].tolist()
        set_longtail = set(ls) & game_cold
        set1 = set1.union(set(ls))
        set2 = set2.union(set_longtail)
        ls_tail.append(len(set_longtail))


    return len(set1)/2675, len(set2)/len(game_cold), np.mean(np.array(ls_tail))