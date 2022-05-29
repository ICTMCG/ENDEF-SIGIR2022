import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='bigru_endef')
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--aug_prob', type=float, default=0.1)
parser.add_argument('--max_len', type=int, default=170)
parser.add_argument('--early_stop', type=int, default=5)
parser.add_argument('--root_path', default='./data/')
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--gpu', default='0')
parser.add_argument('--emb_dim', type=int, default=768)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--save_log_dir', default= './logs')
parser.add_argument('--save_param_dir', default= './param_model')
parser.add_argument('--param_log_dir', default = './logs/param')


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from grid_search import Run
import torch
import numpy as np
import random

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


print('lr: {}; model name: {}; batchsize: {}; epoch: {}; gpu: {}'.format(args.lr, args.model_name, args.batchsize, args.epoch, args.gpu))


config = {
        'use_cuda': True,
        'batchsize': args.batchsize,
        'max_len': args.max_len,
        'early_stop': args.early_stop,
        'root_path': args.root_path,
        'aug_prob': args.aug_prob,
        'weight_decay': 5e-5,
        'model':
            {
            # 'mlp': {'dims': [384, 256, 128], 'dropout': 0.2}
            'mlp': {'dims': [384], 'dropout': 0.2}
            },
        'emb_dim': args.emb_dim,
        'lr': args.lr,
        'epoch': args.epoch,
        'model_name': args.model_name,
        'seed': args.seed,
        'save_log_dir': args.save_log_dir,
        'save_param_dir': args.save_param_dir,
        'param_log_dir': args.param_log_dir
        }

if __name__ == '__main__':
    Run(config = config
        ).main()
