from opt import args

def setup_args(dataset_name="cora"):
    args.dataset = dataset_name
    args.device = "cuda:0"

    if args.dataset == 'cora':
        args.lr = 0.02
        args.hidden = 500
        args.dim_head = 500
        args.num_layers = 1
        args.t = 2

    elif args.dataset == 'citeseer':
        args.lr = 0.02
        args.hidden = 500
        args.dim_head = 500
        args.num_layers = 1

    elif args.dataset == 'amap':
        args.lr = 0.02
        args.hidden = 500
        args.dim_head = 500
        args.num_layers = 1

    elif args.dataset == 'bat':
        args.lr = 0.02
        args.hidden = 500
        args.dim_head = 500
        args.num_layers = 1

    elif args.dataset == 'eat':
        args.lr = 0.02
        args.hidden = 500
        args.dim_head = 500
        args.num_layers = 1

    elif args.dataset == 'uat':
        args.lr = 0.02
        args.hidden = 500
        args.dim_head = 500
        args.num_layers = 1

    # other new datasets
    else:
        args.lr = 0.02
        args.hidden = 500
        args.dim_head = 500
        args.num_layers = 1

    print("---------------------")
    print("dataset: {}".format(args.dataset))
    print("learning rate: {}".format(args.lr))
    print("---------------------")

    return args
