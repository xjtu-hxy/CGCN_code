import torch
import numpy as np
import random
import torch.nn.functional as F
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib

def visualize_tsne(x_tensor, y_tensor, use_pca_first=True, save_path=None):
    if isinstance(x_tensor, torch.Tensor):
        x = x_tensor.detach().cpu().numpy()
    else:
        x = x_tensor

    if isinstance(y_tensor, torch.Tensor):
        y = y_tensor.detach().cpu().numpy()
    else:
        y = y_tensor
    if use_pca_first:
        from sklearn.decomposition import PCA
        n_components = min(50, x.shape[1])
        pca = PCA(n_components=n_components)
        x = pca.fit_transform(x)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    x_2d = tsne.fit_transform(x)
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(y)
    cmap = matplotlib.colormaps['tab10']
    colors = cmap(np.linspace(0, 1, len(unique_labels)))  

    for i, label in enumerate(unique_labels):
        idxs = (y == label)
        plt.scatter(x_2d[idxs, 0], x_2d[idxs, 1], c=[colors[i]], label=f'Class {label}', s=10)

    # plt.title('t-SNE visualization of node embeddings')
    # plt.legend()
    # plt.axis('off')

    if save_path:
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, Flag=0):
    # setup_seed(1)
    # # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    # np.random.seed(cnt)
    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)
    global test_index
    global val_index
    if Flag == 0:
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]


        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        val_index = rest_index[:val_lb]
        data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
        test_index = rest_index[val_lb:]
        data.test_mask = index_to_mask(
            rest_index[val_lb:], size=data.num_nodes)
    else:
        val_index = torch.cat([i[percls_trn:percls_trn+val_lb]
                               for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn+val_lb:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    return data, train_index, test_index, val_index

def test(model, data, adj):
    model.eval()
    x, out, _ = model(data, adj)
    accs, losses, preds = [], [], []

    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = out[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        _, out, loss_1 = model(data, adj)

        preds.append(pred.detach().cpu())
        accs.append(acc)
        losses.append(loss_1.detach().cpu())
    return accs, preds, losses

def normalize_adj(adj, self_loop=True, symmetry=False, device='cpu'):
    """
    normalize the adj matrix
    :param adj: input adj matrix
    :param self_loop: if add the self loop or not
    :param symmetry: symmetry normalize or not
    :return: the normalized adj matrix
    """
    # add the self_loop
    if self_loop:
        adj_tmp = adj + torch.eye(adj.shape[0]).to(device)
    else:
        adj_tmp = adj

    # calculate degree matrix and it's inverse matrix
    d = torch.diag(adj_tmp.sum(0)).to(device)
    d_inv = torch.linalg.inv(d)

    # symmetry normalize: D^{-0.5} A D^{-0.5}
    if symmetry:
        sqrt_d_inv = torch.sqrt(d_inv)
        norm_adj = torch.matmul(torch.matmul(sqrt_d_inv, adj_tmp), sqrt_d_inv)

    # non-symmetry normalize: D^{-1} A
    else:
        norm_adj = torch.matmul(d_inv, adj_tmp)
    return norm_adj

def laplacian_filtering(A, X, t, device='cuda:0'):
    A_tmp = A - torch.diag_embed(torch.diag(A))
    A_norm = normalize_adj(A_tmp, self_loop=True, symmetry=True, device=device)
    I = torch.eye(A.shape[0]).to(device)
    L = I - A_norm
    for i in range(t):
        X = (I - L) @ X
    return X.float(), A_norm

def graph_contrastive_loss(x1, x2, labels, temperature=0.07):
    x1 = F.normalize(x1, dim=1, p=2)
    x2 = F.normalize(x2, dim=1, p=2)

    pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
    sim = torch.matmul(x1, x2.T) / temperature
    sim_mask = torch.zeros_like(sim).bool()
    sim_mask[pos_mask] = True

    pos_sim = sim[pos_mask]
    loss = -torch.log((pos_sim.exp().sum(dim=0))/(sim.exp().sum(dim=0).sum(dim=0)))

    return loss.mean()

def high_confidence(Z, center):
    distance_norm = torch.min(F.softmax(square_euclid_distance(Z, center), dim=1), dim=1).values
    value, _ = torch.topk(distance_norm, int(Z.shape[0] * (1 - 0.8)))
    index = torch.where(distance_norm <= value[-1])[0]

    high_conf_index_v1 = torch.nonzero(index).reshape(-1, )
    high_conf_index_v2 = high_conf_index_v1 + Z.shape[0]
    H = torch.cat([high_conf_index_v1, high_conf_index_v2], dim=0)
    H_mat = np.ix_(H.cpu(), H.cpu())
    return H, H_mat

def compute_node_confidence(X, centers, train_id, Y):

    threshold_distances = torch.cdist(X[train_id], centers)
    threshold_min_distances, threshold_min_indices = threshold_distances.min(dim=1)

    threshold = threshold_min_distances.min(dim=0).values + 0.002
    distances = torch.cdist(X, centers)
    min_distances, min_indices = distances.min(dim=1)
    high_confidence_nodes = (min_distances <= threshold)
    high_confidence_indices = torch.where(high_confidence_nodes)[0]
    high_confidence_label = min_indices[high_confidence_indices]
    return high_confidence_indices, high_confidence_label

def square_euclid_distance(Z, center):
    ZZ = (Z * Z).sum(-1).reshape(-1, 1).repeat(1, center.shape[0])
    CC = (center * center).sum(-1).reshape(1, -1).repeat(Z.shape[0], 1)
    ZZ_CC = ZZ + CC
    ZC = Z @ center.T
    distance = ZZ_CC - 2 * ZC
    return distance

def compute_class_centers(X, Y):
    unique_classes = torch.unique(Y)
    class_centers = torch.zeros((unique_classes.size(0), X.size(1)))
    for i, label in enumerate(unique_classes):
        class_mask = Y == label
        class_features = X[class_mask]
        class_center = class_features.mean(dim=0)
        class_centers[i] = class_center

    return class_centers

def count_equal_elements(tensor1, tensor2):
    equal_mask = tensor1.eq(tensor2)
    equal_count = equal_mask.sum().item()

    # unequal_count = equal_count

    return equal_count

def discriminability(X, Y):
    if X.dim() != 2:
        raise ValueError("X should be a 2D tensor")
    D_squared = F.pairwise_distance(X.unsqueeze(1), X.unsqueeze(0), p=2).pow(2)
    classes, class_counts = torch.unique(Y, return_counts=True)
    numerator = 0.0
    denominator = 0.0

    for i, class_i in enumerate(classes):
        for j, class_j in enumerate(classes):
            if i != j:
                indices_i = torch.where(Y == class_i)[0]
                indices_j = torch.where(Y == class_j)[0]

                inter_class_distance = D_squared[indices_i][:, indices_j].mean()
                numerator += inter_class_distance / (class_counts[i] * class_counts[j])

    for i, class_i in enumerate(classes):
        indices_i = torch.where(Y == class_i)[0]
        intra_class_distance = D_squared[indices_i][:, indices_i].mean()
        denominator += intra_class_distance / (class_counts[i] ** 2)
    discriminability = numerator / denominator
    return discriminability