import os
from setup import setup_args
from utils import *
from dataset_utils import DataLoader
from model import CGCN
from tqdm import tqdm

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(current_dir, 'result.csv')
    if not os.path.exists(result_dir):
        with open(result_dir, 'w') as file:
            file.write("This is a new file.")
    
    for dataset_name in ["cora",'citeseer','dblp', 'acm', 'amac', 'amap']:
        args = setup_args(dataset_name)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        file = open(result_dir, "a+")
        print(args.dataset, file=file)
        print("test_acc, best_val_acc", file=file)
        file.close()

        dataset = DataLoader(dataset_name)
        data = dataset[0]
        # num_nodes = data.x.Size[0]

        # data split
        train_rate = args.train_rate    #0.025
        val_rate = args.val_rate    #0.025
        percls_trn = int(round(train_rate*len(data.y)/dataset.num_classes))
        val_lb = int(round(val_rate*len(data.y)))
        TrueLBrate = (percls_trn*dataset.num_classes+val_lb)/len(data.y)
        print('True Label rate: ', TrueLBrate)

        values = torch.tensor([1 for i in range(data.edge_index.shape[1])])
        adj = torch.sparse.FloatTensor(data.edge_index, values, torch.Size([data.x.shape[0], data.x.shape[0]])).to_dense().cuda()
        adj += torch.eye(adj.shape[0]).long().cuda()
        adj = adj.float()

        Results0 = []
        d_values = []

        for args.seed in range(10):
            setup_seed(args.seed)

            permute_masks = random_planetoid_splits
            data, train_id, val_id, test_id = permute_masks(data, dataset.num_classes, percls_trn, val_lb)  

            data = data.to(device)

            data.x, adj = laplacian_filtering(adj, data.x, args.t, args.device)

            args.num_classes = len(data.y.unique())

            data.train_id = train_id
            data.test_id = test_id
            data.val_id = val_id

            model = CGCN(dataset, args)
            optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)
            model = model.to(device)

            best_val_acc = test_acc = 0
            best_val_loss = float('inf')
            val_loss_history = []
            val_acc_history = []

            best_test_acc = 0
            
            for epoch in tqdm(range(args.epochs)): 
                model.train()
                optimizer.zero_grad()
                x, out, loss_1 = model(data, adj)
                loss = loss_1
                loss.backward()

                optimizer.step()
                # del out

                [train_acc, val_acc, tmp_test_acc], preds, [
                train_loss, val_loss, tmp_test_loss] = test(model, data, adj)

                if tmp_test_acc > test_acc:
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    test_acc = tmp_test_acc

            Results0.append([test_acc, best_val_acc])
            file = open(result_dir, "a+")
            print("{:.2f}, {:.2f}".format(test_acc, best_val_acc), file=file)
            file.close()

        d_values_mean = np.mean(d_values)
        d_values_std = np.sqrt(np.var(d_values)[0])
        print("d_values:{:.2f}, {:.2f}".format(d_values_mean, d_values_std))

        test_acc_mean, val_acc_mean = np.mean(Results0, axis=0) * 100
        
        test_acc_std = np.sqrt(np.var(Results0, axis=0)[0]) * 100
        print(f'test acc mean = {test_acc_mean:.4f} \t test acc std = {test_acc_std:.4f} \t val acc mean = {val_acc_mean:.4f}')
        file = open(result_dir, "a+")
        print("{:.2f}, {:.2f}".format(test_acc_mean, test_acc_std), file=file)
        print("{:.2f}".format(val_acc_mean), file=file)
        file.close()
