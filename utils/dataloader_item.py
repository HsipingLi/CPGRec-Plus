import os
import sys  
from dgl.data.utils import save_graphs
from tqdm import tqdm
from scipy import stats
from utils.NegativeSampler import NegativeSampler
import pdb
import torch
import logging
logging.basicConfig(stream = sys.stdout, level = logging.INFO)
import numpy as np
import dgl
from dgl.data import DGLDataset
import pandas as pd
from sklearn import preprocessing
from dgl.data import DGLDataset
import pickle


class Dataloader_item_graph(DGLDataset):
    def __init__(self, dataloader_steam):

        self.publisher = dataloader_steam.publisher_mapping     
        self.developer =dataloader_steam.developer_mapping    
        self.genre = dataloader_steam.genre_mapping   

        path_dic_genre = "./data_exist/dic_genre.pkl"
        path_dic_pub = "./data_exist/dic_publisher.pkl"
        path_dic_dev = "./data_exist/dic_developer.pkl"

        if not os.path.exists(path_dic_genre) or not os.path.exists(path_dic_pub) or not os.path.exists(path_dic_dev):
            with open(path_dic_genre, 'wb') as f:
                pickle.dump(self.genre, f)
            with open(path_dic_pub, 'wb') as f:
                pickle.dump(self.publisher, f)
            with open(path_dic_dev, 'wb') as f:
                pickle.dump(self.developer, f)



        path = "./data_exist"
        path_graph_and = path + '/graph_and.bin'
        path_graph_or = path + '/graph_or.bin'




        '''graph 1'''
        if os.path.exists(path_graph_and):
            self.graph_and,_ = dgl.load_graphs(path_graph_and)
            self.graph_and = self.graph_and[0]
        
        else:
            self.genre_pub = self.build_edge_and(self.genre, self.publisher)
            self.genre_dev = self.build_edge_and(self.genre, self.developer)
            self.dev_pub = self.build_edge_and(self.developer, self.publisher)

            graph_data_and = {
                ('game', 'co_genre_pub', 'game'): self.genre_pub,
                ('game', 'co_genre_dev', 'game'): self.genre_dev,
                ('game', 'co_dev_pub', 'game'): self.dev_pub
            }

            self.graph_and = dgl.heterograph(graph_data_and)

            dgl.save_graphs(path_graph_and,[self.graph_and])

            



        '''graph 2'''
        if os.path.exists(path_graph_or):
            self.graph_or,_ = dgl.load_graphs(path_graph_or)[0]

        else:
            self.genre_dev_pub = self.build_edge_or(self.genre, self.developer, self.publisher)
            
            graph_data_or = {
                ('game','co_or','game'): self.genre_dev_pub
            }
            self.graph_or = dgl.heterograph(graph_data_or)






    def build_edge_and(self, mapping_genre, mapping_dev):
        src = []
        dst = []
        keys = list(set(mapping_genre.keys()) & set(mapping_dev.keys()))

        for game in keys:
            mapping_genre[game] = set(mapping_genre[game])
            mapping_dev[game] = set(mapping_dev[game])
            
        
        for i in range(len(keys) - 1):
            for j in range(i +1, len(keys)):
                game1 = keys[i]
                game2 = keys[j]
                if len(set(mapping_genre[game1]) & set(mapping_genre[game2])) > 0 and len(set(mapping_dev[game1]) & set(mapping_dev[game2])) > 0:
                    src.extend([game1, game2])
                    dst.extend([game2, game1])
        
        return (torch.tensor(src), torch.tensor(dst))

        


    def build_edge_and_raw(self, mapping):
        src = []
        dst = []
        keys = list(set(mapping.keys()))

        for game in keys:
            mapping[game] = set(mapping[game])
        
        for i in range(len(keys) - 1):
            for j in range(i +1, len(keys)):
                game1 = keys[i]
                game2 = keys[j]
                if len(set(mapping[game1]) & set(mapping[game2])) > 0 :
                    src.extend([game1, game2])
                    dst.extend([game2, game1])
        
        return (torch.tensor(src), torch.tensor(dst))





    def build_edge_or(self, mapping_genre, mapping_dev, mapping_pub):
        src = []
        dst = []
        keys = list(set(mapping_genre.keys()) | set(mapping_dev.keys()) | set(mapping_pub.keys()))

        for game in keys:
            if game in mapping_genre:
                mapping_genre[game] = set(mapping_genre[game])
            else:
                mapping_genre[game] = set()
            if game in mapping_dev:
                mapping_dev[game] = set(mapping_dev[game])
            else:
                mapping_dev[game] = set()
            if game in mapping_pub:
                mapping_pub[game] = set(mapping_pub[game])
            else:
                mapping_pub[game] = set()

        for i in range(len(keys) - 1):
            for j in range(i +1, len(keys)): 
                game1 = keys[i]
                game2 = keys[j]
                if len(set(mapping_genre[game1]) & set(mapping_genre[game2])) > 0 or len(set(mapping_dev[game1]) & set(mapping_dev[game2])) > 0 or len(set(mapping_pub[game1]) & set(mapping_pub[game2])) > 0:
                    src.extend([game1, game2])
                    dst.extend([game2, game1])
        
        return (torch.tensor(src), torch.tensor(dst))
