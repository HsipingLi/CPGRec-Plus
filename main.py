import sys
sys.path.append('../')
import dgl
import dgl.function as fn
import os
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
from utils.dataloader_steam import Dataloader_steam_filtered
from utils.dataloader_item import Dataloader_item_graph
from models.model import Proposed_model
from models.Predictor import Predictor
import pickle
from scipy.stats import boxcox, f

os.chdir('./CPGRec_plus')


def setup_seed(seed=3407):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def get_valid_mask(DataLoader, graph, valid_user):
    path_valid_mask_trail = "./data_exist/valid_mask.pth"
    if os.path.exists(path_valid_mask_trail):
        valid_mask = torch.load(path_valid_mask_trail)
        return valid_mask
    else:
        valid_mask = torch.zeros(len(valid_user), graph.num_nodes('game'))
        for i in range(len(valid_user)):
            user = valid_user[i]
            item_train = torch.tensor(DataLoader.dic_user_game[user])
            valid_mask[i, :][item_train] = 1
        valid_mask = valid_mask.bool()
        torch.save(valid_mask, path_valid_mask_trail)
        return valid_mask



def construct_negative_graph(graph, etype,device):

    utype, _ , vtype = etype
    src, _ = graph.edges(etype = etype)
    src = src.to(device)
    dst = torch.randint(graph.num_nodes(vtype), size = src.shape).to(device)
    return dst, dgl.heterograph({etype: (src, dst)}, num_nodes_dict = {ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes})




def get_coverage(ls_tensor, mapping_genre,mapping_dev,mapping_pub):
    set_genre = set()
    set_dev = set()
    set_pub = set()
    
    ls_genre = []
    ls_dev = []
    ls_pub = []
    

    for i in ls_tensor:

        if int(i) in mapping_genre.keys():
            type_genre = mapping_genre[int(i)]
            set_genre = set_genre.union(set(type_genre))
            ls_genre.extend(type_genre)

        if int(i) in mapping_dev.keys():
            type_dev = mapping_dev[int(i)]
            set_dev = set_dev.union(set(type_dev))
            ls_dev.extend(type_dev)

        if int(i) in mapping_pub.keys():
            type_pub = mapping_pub[int(i)]
            set_pub = set_pub.union(set(type_pub))
            ls_pub.extend(type_pub)

    ls_genre = torch.unique_consecutive(torch.tensor(ls_genre), return_counts=True)[1]
    ls_dev = torch.unique_consecutive(torch.tensor(ls_dev), return_counts=True)[1]
    ls_pub = torch.unique_consecutive(torch.tensor(ls_pub), return_counts=True)[1]

    ls_genre = ls_genre/torch.sum(ls_genre)
    ls_dev = ls_dev/torch.sum(ls_dev)
    ls_pub = ls_pub/torch.sum(ls_pub)

    entro_genre = -torch.sum(ls_genre * torch.log(ls_genre))
    entro_dev = -torch.sum(ls_dev * torch.log(ls_dev))
    entro_pub = -torch.sum(ls_pub * torch.log(ls_pub))

    cov_genre = float(len(set_genre))
    cov_dev = float(len(set_dev))
    cov_pub = float(len(set_pub))


    coverage = cov_genre + cov_dev + cov_pub
    entropy =  entro_genre + entro_dev + entro_pub

    return [cov_genre, cov_dev, cov_pub, coverage, entro_genre, entro_dev, entro_pub, entropy]





def validate(valid_mask, dic, h, ls_k, mapping_genre,mapping_dev,mapping_pub, to_get_coverage = False):

    users = torch.tensor(list(dic.keys())).long()
    user_embedding = h['user'][users]
    game_embedding = h['game']
    rating = torch.mm(user_embedding, game_embedding.t())
    rating[valid_mask] = -float('inf')

    valid_mask = torch.zeros_like(valid_mask)
    for i in range(users.shape[0]):
        user = int(users[i])
        items = torch.tensor(dic[user])
        valid_mask[i, items] = 1

    _, indices = torch.sort(rating, descending = True)
    indices = indices.cpu()
    ls = [valid_mask[i,:][indices[i, :]] for i in range(valid_mask.shape[0])]
    result = torch.stack(ls).float()
    
    res = []
    ndcg = 0
    for k in ls_k:

        discount = (torch.tensor([i for i in range(k)]) + 2).log2()
        ideal, _ = result.sort(descending = True)
        idcg = (ideal[:, :k] / discount).sum(dim = 1)
        dcg = (result[:, :k] / discount).sum(dim = 1)
        ndcg = torch.mean(dcg / idcg)



        recall = torch.mean(result[:, :k].sum(1) / result.sum(1))
        hit = torch.mean((result[:, :k].sum(1) > 0).float())
        precision = torch.mean(result[:, :k].mean(1))


        if to_get_coverage == False:
            coverage = -1
        else:
            cover_tensor = torch.tensor([get_coverage(indices[i,:k],mapping_genre,mapping_dev,mapping_pub) for i in range(users.shape[0])])
            cov_genre, cov_dev, cov_pub, coverage, entro_genre, entro_dev, entro_pub, entro = torch.mean(cover_tensor, axis = 0)


        logging_result = "For k = {}, ndcg = {}, recall = {}, hit = {}, precision = {}, cov_genre = {}, cov_dev = {}, cov_pub = {}, coverage = {}, entro_genre = {}, entro_dev = {}, entro_pub = {}, entro = {}".format(k, ndcg, recall, hit, precision, cov_genre, cov_dev, cov_pub, coverage, entro_genre, entro_dev, entro_pub, entro)
        logging.info(logging_result)
        res.append(logging_result)
    return  coverage, str(res)








def concate_rating(DataLoader, path_dic_app_info):
    
    path_inter_concated = "./data_exist/user_game_concated.pth"
    if os.path.exists(path_inter_concated):
        return torch.load(path_inter_concated)


    else:
        with open(path_dic_app_info, "rb") as f:
            dic_app_info = pickle.load(f)

        inter = DataLoader.user_game
        inter = torch.hstack([inter, torch.ones(size=(inter.shape[0], 1))])

        for game in tqdm(dic_app_info.keys()):
            index = (inter[:,1]==game)
            inter[index,-1:] = torch.tensor(dic_app_info[game][-1:].astype(float))*torch.ones_like(inter[index, -1:])              

        means = torch.nanmean(inter, dim=0)
        inter = torch.where(torch.isnan(inter), means, inter)
        torch.save(inter, path_inter_concated)
        return inter



def get_fstat(path_dic_app_info, user_game):


    t, r = user_game[:,2], user_game[:,3]
    t, r = t-t.min(), r-r.min()
    t, r = torch.tensor(boxcox(t+1e-6)[0]), torch.tensor(boxcox(r+1e-6)[0])
    t, r = (t - t.mean())/t.std(), (r - r.mean())/r.std()
    
    f_stat = t**2/r**2
    torch.save(f_stat, path_fstat)
    return f_stat, t, r






if __name__ == '__main__':

    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    args.gpu = 0


#Load Data

    DataLoader = Dataloader_steam_filtered(args.path, device = device)
    graph = DataLoader.graph.to(device)
    DataLoader_item = Dataloader_item_graph(DataLoader)
    
    graph_item_and = DataLoader_item.graph_and
    graph_item_or = DataLoader_item.graph_or
    graph = dgl.edge_type_subgraph(graph, [('user','play','game'),('game','played by','user')])

    valid_user = list(DataLoader.valid_data.keys())
    valid_mask = get_valid_mask(DataLoader, graph, valid_user)




#Prepare Weight of PER Module
    path_fstat="./data_exist/fstat.pth"
    path_t = "./data_exist/t.pth"
    path_r = "./data_exist/r.pth"

    if os.path.exists(path_fstat):
        f_stat = torch.load(path_fstat)
        t, r = torch.load(path_t), torch.load(path_r)
        t, r = (t - t.mean())/t.std(), (r - r.mean())/r.std()
    
    else:
        path_dic_app_info = "./data_exist/dic_app_info.pkl"
        DataLoader.user_game = concate_rating(DataLoader, path_dic_app_info)
        user_game = DataLoader.user_game
        f_stat, t, r = get_fstat(path_dic_app_info, user_game)
        torch.save(f_stat, path_fstat)
    

    alpha = 0.4
    Q_l, Q_u = f.ppf(alpha, 1, 1), f.ppf(1-alpha, 1, 1)

    mask_sig = f_stat >= Q_u
    mask_like, mask_hate = t>r, t<=r
    mask_like, mask_hate = mask_like & mask_sig, mask_hate & mask_sig
    mask_mode = ~ (mask_like | mask_hate)

    # mask_pos, mask_neg = f_stat >= Q_u, f_stat <= Q_l
    # mask_one = ~ (mask_pos | mask_neg)



    def norm_pdf(x):
        return (1 / torch.sqrt(2 * torch.tensor(torch.pi))) * torch.exp(-x**2 / 2)
    ft, fr = norm_pdf(t), norm_pdf(r)
    info_cont = ((ft * fr)+1e-6).log() * (-1)

    weight_PER = torch.ones_like(info_cont)
    weight_PER[mask_like] = info_cont[mask_like]
    weight_PER[mask_hate] = info_cont[mask_hate] * (-1)
    weight_PER[mask_mode] = torch.zeros(mask_mode.sum(),dtype=torch.float64)

    # weight_PER = None

#Init Model

    model = Proposed_model(args, graph, graph_item_and, graph_item_or, device, weight_PER)
    model.to(device)    




#Train

    predictor = Predictor()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    stop_count = 0
    ls_k = [5, 10]
    total_epoch = 0
    
    loss_pre = 0
    loss = 0

    for epoch in range(args.epoch):
        model.train()
        dst, graph_neg = construct_negative_graph(graph,('user','play','game'),device)
        h, h_and, h_or = model()

        score = predictor(graph, h, ('user','play','game'),device)
        score_neg = predictor(graph_neg, h, ('user','play','game'),device)
        loss_pre = loss

        
        score_neg_reweight = score_neg * (score_neg.sigmoid()*args.m)

        loss =  ((-((score - score_neg_reweight).sigmoid().clamp(min=1e-8, max=1-1e-8).log()[:,0]))).sum()

        
        loss = loss.to(device)
        logging.info('Epoch {}'.format(epoch))
        logging.info(f"loss = {loss}\n")
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_epoch += 1
        

        if (total_epoch > 1 and total_epoch%50==0) or (total_epoch>=900 and total_epoch%10==0):
            model.eval()
            validate(valid_mask, DataLoader.test_data, h, ls_k, DataLoader_item.genre,DataLoader_item.developer,DataLoader_item.publisher, to_get_coverage = True)
            if loss < loss_pre:
                stop_count = 0
            else:
                stop_count += 0
                logging.info(f"stop count:{stop_count}")
                if stop_count > args.early_stop:
                    logging.info('early stop')
                    break

        
        if epoch==args.epoch-1:
            validate(valid_mask, DataLoader.test_data, h, ls_k, DataLoader_item.genre,DataLoader_item.developer,DataLoader_item.publisher, to_get_coverage = True)