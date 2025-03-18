import torch.nn as nn
import dgl.function as fn
import torch
class Predictor(nn.Module):
    def forward(self, graph, h, etype,device):
        with graph.local_scope():


            cpu = torch.device("cpu")
            gpu = torch.device(device)
            
            if h['user'].device == gpu and graph.device == cpu:
                graph = graph.to(gpu)
            elif h['user'].device == cpu and graph.device == gpu:
                h['user'] = h['user'].to(gpu)
                h['game'] = h['game'].to(gpu)
            
                
            

            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype = etype)
            return graph.edges[etype].data['score']
