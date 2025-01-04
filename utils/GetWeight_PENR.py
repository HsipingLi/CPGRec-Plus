import dgl
import torch



def get_penr_weight(theta1=80, theta2=0.5, theta3=3):
    
    torch.manual_seed(2023)
    path = "./data_exist/graph.bin"

    graph = dgl.load_graphs(path)[0]
    graph = graph[0]

    graph = dgl.edge_type_subgraph(graph, etypes=['played by'])

    outdeg_game = graph.out_degrees(graph.edges(etype = ('game','played by','user'))[0],etype = 'played by').float()


    theta1 =  80 #enhanced edge weight hyper parameter
    theta2 = 0.5    #weakend node weight hyper parameter
    theta3 = 3  #enhanced node weight hyper parameter


    quantile_low = 50000    #50000: quantile(outdeg_game, 0.2000)
    quantile_high = 850000  #850000: quantile(outdeg_game, 0.8000)

    idx_low = (outdeg_game<=quantile_low)
    idx_high = (outdeg_game>quantile_high)
    torch.save(idx_high, "./data_exist/idx_high.pth")

    weight_edge = torch.ones_like(outdeg_game)
    weight_edge[idx_high] = weight_edge[idx_high] * theta1


    path_weight_edge = "./data_exist/weight_edge.pth"
    torch.save(weight_edge, path_weight_edge)



    outdeg_game_2 = graph.out_degrees( graph.nodes(ntype = 'game'),etype = 'played by').float()
    weight_node = torch.ones_like(outdeg_game_2)

    quantile_low_2 = 570    #570: quantile(outdeg_game_2, 0.2000)
    quantile_high_2 = 33000 #33000: quantile(outdeg_game_2, 0.8000)

    path_cold =  "./data_exist/game_cold.pth"
    path_hot =  "./data_exist/game_hot.pth"
    torch.save(torch.where(outdeg_game_2 <= quantile_low_2), path_cold)
    torch.save(torch.where(outdeg_game_2 >= quantile_high_2), path_hot)


    idx_low = (outdeg_game_2<=quantile_low_2)
    idx_high = (outdeg_game_2>quantile_high_2)
    weight_node[idx_low] = weight_node[idx_low] * theta3
    weight_node[idx_high] = weight_node[idx_high] * theta2

    path_weight_node = "./data_exist/weight_node.pth"
    torch.save(weight_node, path_weight_node)

    return weight_edge, weight_node
