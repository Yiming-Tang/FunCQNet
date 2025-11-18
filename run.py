import argparse
import os
import torch
import torch.backends
from exp.exp_main import Exp_main
from utils.tools import pad_and_convert_to_dict
from utils.tools import save_print
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader
from torch.utils.data import Subset
import random
import numpy as np
import pandas as pd

if __name__ == '__main__':

    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    
    parser = argparse.ArgumentParser(description='DFQR')   #  Deep Functional Quantile Regression with Censoring and Structured Interactions
    
    # basic config
    parser.add_argument('--model', type=str, default='DFQR',
                        help='model name, options: [DFQR, DFQRwoI]')
    
    # data loader
    parser.add_argument('--data', type=str, default='S1_1_c03_01_data0817', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/s1/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='S1_1_c03_01_data0815_eps0.01.pt', help='data file')
    parser.add_argument('--save_path', type=str, default='./save_models/', help='location of saved models')
    parser.add_argument('--maxlen_function', type=int, default=None, help='max length of functional data')
    
    
    # quantile define
    parser.add_argument('--num_quantiles', type=int, default=99, help='number of quantiles, e.g. 9 or 99')
    
    # model define
    parser.add_argument('--neurons', type=int, nargs='+', default=[64, 128, 64], help='number of neurons in each layer')
    parser.add_argument('--dropout', type=float, default=0.01, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    
    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=1000, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    
    
    
    
    args = parser.parse_args()             #save this in .py file
    #args = parser.parse_args(args=[])       #save this in .ipynb file
    
    setting = '{}_{}_{}_{}_{}'.format(
        args.model,
        args.data,
        args.num_quantiles,
        args.learning_rate,
        args.neurons)
    save_print("./log", setting)
    
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')
    
    args.device = torch.device("cuda:0")     # specify device
    
    print('Args in experiment:')
    #print_args(args)
    
    
    df_raw = torch.load(os.path.join(args.root_path, args.data_path))
    
    if isinstance(df_raw, list):
        # If the input is a list, perform padding and convert to dictionary
        df_raw, final_len = pad_and_convert_to_dict(df_raw, target_len=args.maxlen_function)  # target_len is optional
        print(f"Converted list to dictionary format with uniform length, final length is {final_len}")
    else:
        print("Data is already in dictionary format, using directly")
        
    n = len(df_raw['X_all']) 
    I_all = torch.ones(n,1)
    train_size = int(0.7*n)
    val_size = int(0.1*n)
    test_size = n - train_size - val_size
    
    args.p = df_raw['X_all'].shape[-1]
    args.q = df_raw['Z_all'].shape[-1]
    
    dataset = torch.utils.data.TensorDataset(I_all[0:n], df_raw['X_all'][0:n], df_raw['t_all'][0:n], df_raw['Z_all'][0:n], df_raw['Y_all'][0:n], df_raw['C_all'][0:n])
    train_dataset = Subset(dataset, range(0, train_size))
    val_dataset = Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = Subset(dataset, range(train_size + val_size, n))
    
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset,batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset,batch_size=args.batch_size, shuffle=False)
    print(f"Train set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}, Test set size: {len(test_dataset)}")
    
    Y_pred_all = torch.zeros((n, 1))
    for tau in range(args.num_quantiles):
        args.tau = (tau + 1) / (args.num_quantiles + 1)
        
        dataset_pred = torch.utils.data.TensorDataset(Y_pred_all)
        train_pred = Subset(dataset_pred, range(0, train_size))
        val_pred = Subset(dataset_pred, range(train_size, train_size + val_size))
        test_pred = Subset(dataset_pred, range(train_size + val_size, n))
    
        train_loader_pred = DataLoader(train_pred,batch_size=args.batch_size, shuffle=False)
        val_loader_pred = DataLoader(val_pred,batch_size=args.batch_size, shuffle=False)
        test_loader_pred = DataLoader(test_pred,batch_size=args.batch_size, shuffle=False)
    
        # setting record of experiments
        exp = Exp_main(args, train_loader, val_loader, test_loader, 
            train_loader_pred, val_loader_pred, test_loader_pred)  # set experiments
        setting = '{}_{}_{}_{}_{}'.format(
            args.model,
            args.data,
            args.num_quantiles,
            args.learning_rate,
            args.neurons)
        
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        print(f"tau={args.tau}")
        exp.train(setting)
        
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        
        Y_pred_all = torch.cat([Y_pred_all.cpu(), exp.updateY_pred(setting, I_all, df_raw['X_all'][0:n], df_raw['t_all'][0:n], df_raw['Z_all'][0:n])], dim=1)
        
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()

