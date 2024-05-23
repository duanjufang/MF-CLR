import argparse
import sys
import os
abs_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(abs_path)
import torch
from exp.informer import Exp_Informer
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data_loader import Dataset_Custom, Dataset_Pred
Exp = Exp_Informer

def default_config():
    ## get default setting
    parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

    parser.add_argument('--model', type=str, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')

    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    parser.add_argument('--seq_len', type=int, default=6, help='input sequence length of Informer encoder')
    parser.add_argument('--label_len', type=int, default=4, help='start token length of Informer decoder')
    parser.add_argument('--pred_len', type=int, default=2, help='prediction sequence length')
    # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

    parser.add_argument('--enc_in', type=int, default=3, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
    parser.add_argument('--padding', type=int, default=0, help='padding type')
    parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu',help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test',help='exp description')
    parser.add_argument('--loss', type=str, default='mse',help='loss function')
    parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    return args

def get_data(data_sequence, flag, args, y_dim):
    
    Data = Dataset_Custom

    if flag == 'test':
        shuffle_flag = False; drop_last = True; batch_size = args.batch_size; 
    elif flag=='pred':
        shuffle_flag = False; drop_last = False; batch_size = 1; 
        Data = Dataset_Pred
    else:
        shuffle_flag = True; drop_last = True; batch_size = args.batch_size; 
        
    data_set = Data(
            data_sequence,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            inverse=args.inverse,
            y_dim= y_dim
        )
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        )
    return data_set, data_loader


def train_and_predict(data_sequence: np.array, window_len, pred_len, enc_in, y_dim):
    """
    a pipeline for train and predict
    data_sequence: (n), time series for train and predict
    """
    args = default_config()
    args.seq_len = window_len
    args.pred_len = pred_len
    args.enc_in = enc_in
    # 1. definition model
    exp = Exp(args)
    
    #2. dataset
    train_data, train_loader = get_data(data_sequence, 'train', args, y_dim)
#     vali_data, vali_loader = get_data(data_sequence, 'val', args)
#     test_data, test_loader = get_data(data_sequence, 'test', args)
    pred_data, pred_loader = get_data(data_sequence, 'pred', args, y_dim)
    
    #3. train
    print('>>>>>>>start training>>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.train(train_data, train_loader)
    print('>>>>>>>testing<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
#     exp.test(test_data, test_loader)
    #4.predict and return
    print('>>>>>>>predicting<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    results = exp.predict(pred_data, pred_loader)
    print("results.shape : ", results.shape)
    print(results)
    ss = pred_data.scaler
    # results = np.around(ss.inverse_transform(results).reshape([-1]))
    # results = np.around(ss.inverse_transform(results)[ : , : , y_dim].reshape(-1))
    results = ss.inverse_transform(results, y_dim).reshape(-1)
    return results
    
if __name__ == '__main__':
    import pandas as pd
    data_sequence = np.array([6,5,4,3,2,1]*1000).reshape(-1, 12)
    results = train_and_predict(data_sequence, 13, 2, 2)
    print(data_sequence[ -4 : , : ])
    print(results)