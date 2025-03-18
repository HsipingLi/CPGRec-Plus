import torch.nn as nn
from tqdm import tqdm
import torch
import dgl
import dgl.function as fn
import dgl.nn as dglnn
from dgl.nn.pytorch.conv import GraphConv
import torch.nn.functional as F
import pdb
from utils.GetWeight_PENR import get_penr_weight
from utils.item_prompt_Qwen import get_item_embedding_with_LLM

class MLP(nn.Module):
    def __init__(self, in_feats, hid_dim, out_feats, num_layers=2, drop_rate=0, activation='ReLU'):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        if num_layers == 0:
            return
        if num_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hid_dim))
            for i in range(1, num_layers-1):
                self.layers.append(nn.Linear(hid_dim, hid_dim))
            self.layers.append(nn.Linear(hid_dim, out_feats))
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, h):
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h)
            if i != len(self.layers)-1:
                h = self.act(h)
        return h




class Proposed_model(nn.Module):
    def __init__(self, args, graph, graph_and, graph_or, device, weight_PER=None):
        super().__init__()
        self.device = torch.device(device)
        self.args = args
        self.param_decay = args.param_decay
        self.hid_dim = args.embed_size  
        self.attention_and = args.attention_and
        self.layer_num_and = args.layers_and  
        self.layer_num_or = args.layers_or 
        self.layer_num_user_game = args.layers_user_game
        self.gamma = args.gamma
        self.graph_and = graph_and.to(self.device)
        self.graph_or = graph_or.to(self.device)
        self.graph = graph.to(self.device)
        self.graph_item2user = dgl.edge_type_subgraph(self.graph,['played by']).to(self.device)
        self.graph_user2item = dgl.edge_type_subgraph(self.graph,['play']).to(self.device)
        self.weight_edge_PER = weight_PER.to(self.device) if weight_PER is not None else None
        

        
        self.weight_edge_PENR, self.weight_node_PENR = get_penr_weight()
        self.weight_edge_PENR, self.weight_node_PENR = self.weight_edge_PENR.to(self.device), self.weight_node_PENR.to(self.device)


        self.w_or = self.gamma / (self.gamma + 2)
        self.w_and = self.w_or / self.gamma
        self.w_self = self.w_or / self.gamma


        self.user_embedding = torch.nn.Parameter(torch.randn(self.graph.nodes('user').shape[0], self.hid_dim)).to(torch.float32)
        self.item_embedding = torch.nn.Parameter(torch.randn(self.graph.nodes('game').shape[0], self.hid_dim)).to(torch.float32)


        self.W_and = torch.nn.Parameter(torch.randn(self.args.embed_size, self.args.embed_size)).to(torch.float32)
        self.a_and = torch.nn.Parameter(torch.randn(self.args.embed_size)).to(torch.float32)
        self.W_or = torch.nn.Parameter(torch.randn(self.args.embed_size, self.args.embed_size)).to(torch.float32)
        self.a_or = torch.nn.Parameter(torch.randn(self.args.embed_size)).to(torch.float32)

        self.conv = GraphConv(self.hid_dim, self.hid_dim, weight=False, bias=False, allow_zero_in_degree = True).to(self.device)
        

        self.build_model_and(self.graph_and)
        self.build_model_or()
        self.build_model_user_game()


        ##updated

        self.user_embedding_LLM = torch.nn.Parameter(torch.randn(self.graph.nodes('user').shape[0], self.hid_dim)).to(torch.float32)
        self.item_embedding_LLM = torch.nn.Parameter(get_item_embedding_with_LLM().to(torch.float32).to(self.device)).to(torch.float32)


        # self.MLP_for_user_LLM = MLP(in_feats=self.user_embedding_LLM.shape[1], hid_dim=64, out_feats=self.hid_dim).to(self.device)
        self.MLP_for_user_final = MLP(in_feats=2*self.hid_dim, hid_dim=self.hid_dim, out_feats=self.hid_dim).to(self.device)
        self.MLP_for_item_LLM = MLP(in_feats=self.item_embedding_LLM.shape[1], hid_dim=64, out_feats=self.hid_dim).to(self.device)
        self.MLP_for_item_final = MLP(in_feats=2*self.hid_dim, hid_dim=self.hid_dim, out_feats=self.hid_dim).to(self.device)

    '''attention'''
    def layer_attention(self, ls, W, a):

        ls = [x.to(self.device) for x in ls]
        tensor_layers = torch.stack(ls, dim=0)
        weight = torch.matmul(tensor_layers, W)
        weight = F.softmax(weight*a, dim=0)
        tensor_layers = torch.sum(tensor_layers * weight, dim=0)
        return tensor_layers




    '''model for graph_and'''

    def build_model_and(self, graph_and):

        self.sub_g1 = dgl.edge_type_subgraph(graph_and,['co_genre_pub']).to(self.device)
        self.sub_g2 = dgl.edge_type_subgraph(graph_and,['co_genre_dev']).to(self.device)
        self.sub_g3 = dgl.edge_type_subgraph(graph_and,['co_dev_pub']).to(self.device)



    def get_h_and(self,attention):
        ls = [self.item_embedding]
        h1 = self.item_embedding
        h2 = self.item_embedding
        h3 = self.item_embedding

        for _ in range(self.layer_num_and):
            h1 = self.conv(self.sub_g1.cpu(), h1.cpu())
            h2 = self.conv(self.sub_g2.cpu(), h2.cpu())
            h3 = self.conv(self.sub_g3.cpu(), h3.cpu())
            ls.append(h1.to(self.device))
            ls.append(h2.to(self.device))
            ls.append(h3.to(self.device))

        
        
        if attention == True:
            return self.layer_attention(ls, self.W_and.to(self.device), self.a_and.to(self.device))
        else:   
            return ((h1+h2+h3)/3).cpu()



    '''model for graph_or'''

    def build_model_or(self):
        self.model_list_or = nn.ModuleList()
        for _ in range(self.layer_num_or):
            self.model_list_or.append(GraphConv(self.hid_dim, self.hid_dim, weight = False, bias = False, allow_zero_in_degree = True).to(self.device))


    def get_h_or(self, graph_or):
        ls = [self.item_embedding.cpu()]
        h_temp = self.item_embedding.cpu()
        layer_idx = 1
        param_min = 0.2
        
        for layer in self.model_list_or:
            param = 1-(self.layer_num_or-layer_idx)*self.param_decay
            layer_idx += 1
            param = max(list([param, param_min]))

            h_temp = layer(graph_or.cpu(), h_temp.cpu())
            ls.append((h_temp * param).to(self.device))

        return self.layer_attention(ls, self.W_or.to(self.device), self.a_or.to(self.device))




    def build_model_user_game(self):
        self.layers = nn.ModuleList()
        for _ in range(self.layer_num_user_game):
            layer = GraphConv(self.hid_dim,self.hid_dim, weight=False, bias=False, allow_zero_in_degree = True)

            self.layers.append(layer)
        self.layers.to(self.device)
    


    '''forward'''
    def forward(self):

        h = {'user':self.user_embedding.to(self.device), 'game':self.item_embedding.to(self.device)}

        h_user_LLM = self.user_embedding_LLM    #*
        
        # h_user_LLM = self.MLP_for_user_LLM(self.user_embedding_LLM)  #* 
        h_item_LLM = self.MLP_for_item_LLM(self.item_embedding_LLM)

        h['user'] = self.MLP_for_user_final(torch.hstack([h['user'], h_user_LLM]))    #*
        h['game'] = self.MLP_for_item_final(torch.hstack([h['game'], h_item_LLM]))


        for layer in self.layers:
            h['game'] = torch.matmul(torch.diag(self.weight_node_PENR.to(self.device)) , h['game'] )
            try:
                if self.weight_edge_PER is not None:
                    h_user = layer(self.graph_item2user, (h['game'],h['user']),edge_weight=self.weight_edge_PENR + self.weight_edge_PER)
                else:
                    h_user = layer(self.graph_item2user, (h['game'],h['user']),edge_weight=self.weight_edge_PENR)
            except:
                h_user = layer(self.graph_item2user, (h['game'],h['user']),edge_weight=self.weight_edge_PENR)
            try:
                h_item = layer(self.graph_user2item, (h['user'],h['game']))
            except:
                h_item = layer(self.graph_user2item.cpu(), (h['user'].cpu(),h['game'].cpu())).to(self.device)

            h['user'] = h_user
            h['game'] = h_item



        h_and = self.get_h_and(attention=self.attention_and)
        h_or = self.get_h_or(self.graph_or)
        h_self = h['game']

        h['game'] = self.w_and * h_and + self.w_or * h_or  + self.w_self * h_self
        return h, h_and, h_or
