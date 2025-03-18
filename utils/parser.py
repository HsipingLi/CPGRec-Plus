import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default = 3407, type = int)
    parser.add_argument('--embed_size', default = 32, type = int)
    parser.add_argument('--lr', default = 0.03, type = float)
    parser.add_argument('--epoch', default = 1000, type = int)
    parser.add_argument('--early_stop', default = 10, type = int)
    parser.add_argument('--gpu', default = 0, type = int)
    parser.add_argument('--k', default = [5,10], type = list)
    parser.add_argument('--m', default = 4, type = float)
    parser.add_argument('--w_Co', default =80/82, type = float)
    parser.add_argument('--w_Ca', default = 1/82, type = float)
    parser.add_argument('--w_Po', default = 1/82, type = float)
    parser.add_argument('--gamma', default=80.0, type=float)
    parser.add_argument('--layers_and', default=2, type=int)
    parser.add_argument('--layers_or', default=4, type=int)
    parser.add_argument('--layers_user_game', default=2, type=int)
    parser.add_argument('--attention_and', default=True, type=bool)
    parser.add_argument('--param_decay', default=0.1, type=float)
    parser.add_argument('--path', default='./steam_data', type=str)

    
    args = parser.parse_args()
    return args
