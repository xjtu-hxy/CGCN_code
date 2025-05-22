import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--lr', type=float, default=0.02)
parser.add_argument('--weight_decay', type=float, default=0.0005)

parser.add_argument('--early_stopping', type=int, default=200)
parser.add_argument('--dropout', type=float, default=0.7)

parser.add_argument('--train_rate', type=float, default=0.025)
parser.add_argument('--val_rate', type=float, default=0.025)

parser.add_argument('--hidden', default=500, type=int)
parser.add_argument('--dim_head', default=500, type=int)
parser.add_argument('--num_layers', default=1, type=int)

parser.add_argument('--dataset', default='cora')
parser.add_argument('--device', type=str, default='cuda:0')

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_classes', type=int, default=7)
parser.add_argument('--num_nodes', type=int, default=2708)

parser.add_argument('--t', type=int, default=2)

args = parser.parse_args()