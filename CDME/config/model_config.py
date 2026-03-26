import os 
import argparse

def build_args(dataset="VisDA"):
    
    parser = argparse.ArgumentParser("This script is used to SFDA")
    parser.add_argument("--dataset", type=str, default="VisDA")
    parser.add_argument("--backbone_arch", type=str, default="resnet101", help="restnet50, resnet101, vgg")
    parser.add_argument("--embed_feat_dim", type=int, default=256)
    parser.add_argument("--s_idx", type=int, default=0)
    parser.add_argument("--t_idx", type=int, default=1)
    parser.add_argument("--distance", default="cosine", type=str)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--without_wandb", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--note", default="None", type=str)
    parser.add_argument("--seed", default=2021, type=int)
    parser.add_argument("--multi_cent_num", default=4, type=int)
    parser.add_argument("--topk_seg", default=3, type=int)
    parser.add_argument("--lam_psd", default=0.30, type=float)
    parser.add_argument("--lam_dym", default=0.10, type=float)
    parser.add_argument("--lam_reg", default=1.0, type=float)
    parser.add_argument("--lam_ent", default=1.0, type=float)
    parser.add_argument("--lam_proto", type=float, default=0.2,
                        help="weight for prototype contrastive loss")
    parser.add_argument("--dym_global_scale", action="store_true")
    parser.add_argument("--dym_tau_clip_low", type=float, default=0.85)
    parser.add_argument("--dym_tau_clip_high", type=float, default=1.20)
    args = parser.parse_args()
    # args.dataset = dataset
    
    if args.dataset == "VisDA":
        args.source_data_dir = "./data/VisDA/train/"
        args.target_data_dir = "./data/VisDA/validation/"
        args.lr = 1e-3
        args.class_num = 12
        args.multi_cent_num = 4

        args.lam_psd = 2  # α
        args.lam_dym = 0.5  # β

        args.tau_min = 0.6  # ★ 新增：大数据集 τ 区间偏大一点
        args.tau_max = 1.2

        args.lam_proto = 0.2

        args.dym_global_scale = True
        args.dym_tau_clip_low = 0.85
        args.dym_tau_clip_high = 1.20
        if args.backbone_arch == "auto" or args.backbone_arch == "resnet101":
            args.backbone_arch = "resnet101"
        
    elif args.dataset == "OfficeHome":
        # OfficeHome dataset need to generate img_list.
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.source_data_dir = os.path.join("./data/OfficeHome", names[args.s_idx])
        args.target_data_dir = os.path.join("./data/OfficeHome", names[args.t_idx])
        args.lr = 1e-2
        args.class_num = 65
        args.multi_cent_num = 4

        args.lam_psd = 0.3  # α
        args.lam_dym = 0.1

        args.tau_min = 0.8  # ★ τ 区间收窄
        args.tau_max = 1.1

        args.lam_proto = 0.05

        args.dym_global_scale = False
        args.dym_tau_clip_low = 0.95
        args.dym_tau_clip_high = 1.05
        if args.backbone_arch == "auto" or args.backbone_arch == "resnet101":
            args.backbone_arch = "resnet50"
        
    elif args.dataset == "Office":
        names = ['Amazon', 'Dslr', 'Webcam']
        args.source_data_dir = os.path.join("./data/Office", names[args.s_idx])
        args.target_data_dir = os.path.join("./data/Office", names[args.t_idx])
        args.lr = 1e-2
        args.class_num = 31
        args.multi_cent_num = 2

        args.lam_psd = 2  # α
        args.lam_dym = 0.1

        args.tau_min = 0.6#0.8
        args.tau_max = 1.2#1.1

        args.lam_proto = 0.05

        args.dym_global_scale = False
        args.dym_tau_clip_low = 0.95
        args.dym_tau_clip_high = 1.05
        if args.backbone_arch == "auto" or args.backbone_arch == "resnet101":
            args.backbone_arch = "resnet50"

    elif args.dataset == "DomainNet":
        names = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        args.source_data_dir = os.path.join("./data/DomainNet", names[args.s_idx])
        args.target_data_dir = os.path.join("./data/DomainNet", names[args.t_idx])

        args.lr = 1e-3
        args.class_num = 345  # ✔ 正确，不是6

        # 超参数（可根据你的方法调整）
        args.multi_cent_num = 4
        args.lam_psd = 2
        args.lam_dym = 0.5
        args.tau_min = 0.6
        args.tau_max = 1.2
        args.lam_proto = 0.2

        args.dym_global_scale = True
        args.dym_tau_clip_low = 0.85
        args.dym_tau_clip_high = 1.20

        if args.backbone_arch == "auto" or args.backbone_arch == "resnet101":
            args.backbone_arch = "resnet101"


    elif args.dataset == "PointDA":
        args.lam_psd = 2.0
        args.lam_dym = 0.5

    else:
        raise ValueError("Wrong Dataset Name!!!")

    if args.dataset in ["OfficeHome", "Office"]:
        if args.lr == 1e-3:
            args.lr = 1e-2
        if args.epochs == 30:
            args.epochs = 50
    if args.test:
        args.without_wandb = True
    
    return args
    