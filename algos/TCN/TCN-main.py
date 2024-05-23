import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os
from scipy import stats
from tqdm import trange
import logging
import argparse
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
torch.set_printoptions(threshold=np.inf)

import utils
from dataloader import *

abs_path = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--is-training', default='True', help='Whether to retrain the model')
parser.add_argument('--origin-datasets', default='PUBLIC_DATASETS', help='Parent dir of the origin datasets')
parser.add_argument('--origin-dataset', default='ETTm1_processed_OT=quarterly.csv', help='Name of the origin dataset')
parser.add_argument('--data-folder', default='data', help='Parent dir of the dataset')
parser.add_argument('--dataset-processed', default='elect', help='Name of the dataset')
parser.add_argument('--model-name', default='TCN', help='Directory containing params.json')
parser.add_argument('--pred-length', default=48, help='Prediction length for each tasks')
parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--sampling', action='store_true', help='Whether to sample during evaluation')
parser.add_argument('--save-best', action='store_true', help='Whether to save best ND to param_search.txt')
parser.add_argument('--restore-file', default='best',
                    help='Optional, name of the file in --model_dir containing weights to reload before \
                    training')  # 'best' or 'epoch_#'


def findByRow(mat, row):
    for i, item in enumerate(mat):
        if list(item) == row:
            return i
    return False

    
### Related functions
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1, init=True):
        super(TemporalBlock, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.init = init
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        if self.init:
            nn.init.normal_(self.conv1.weight, std=1e-3)
            nn.init.normal_(self.conv2.weight, std=1e-3)

            self.conv1.weight[:, 0, :] += (1.0 / self.kernel_size)  ###new initialization scheme
            self.conv2.weight += 1.0 / self.kernel_size  ###new initialization scheme

            nn.init.normal_(self.conv1.bias, std=1e-6)
            nn.init.normal_(self.conv2.bias, std=1e-6)
        else:
            nn.init.xavier_uniform_(self.conv1.weight)
            nn.init.xavier_uniform_(self.conv2.weight)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.1)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalBlock_last(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, init=True):
        super(TemporalBlock_last, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.dropout1, self.conv2, self.chomp2, self.dropout2)
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.init = init
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        if self.init:
            nn.init.normal_(self.conv1.weight, std=1e-3)
            nn.init.normal_(self.conv2.weight, std=1e-3)

            self.conv1.weight[:, 0, :] += (1.0 / self.kernel_size)  ###new initialization scheme
            self.conv2.weight += 1.0 / self.kernel_size  ###new initialization scheme

            nn.init.normal_(self.conv1.bias, std=1e-6)
            nn.init.normal_(self.conv2.bias, std=1e-6)
        else:
            nn.init.xavier_uniform_(self.conv1.weight)
            nn.init.xavier_uniform_(self.conv2.weight)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.1)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + res

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.1, init=True):
        super(TemporalConvNet, self).__init__()
        layers = []
        self.num_channels = num_channels
        self.num_inputs = num_inputs
        self.kernel_size = kernel_size
        self.dropout = dropout
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            if i == num_levels - 1:
                layers += [
                    TemporalBlock_last(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                        padding=(kernel_size - 1) * dilation_size, dropout=dropout, init=init)
                ]
            else:
                layers += [
                    TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                        padding=(kernel_size - 1) * dilation_size, dropout=dropout, init=init)
                ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    

def prep_data(data, covariates, data_start, train=True):
    # print("train: ", train)
    time_len = data.shape[0]
    # print("time_len: ", time_len)
    input_size = window_size-stride_size
    if train:
        windows_per_series = np.full((num_series), (time_len-input_size) // stride_size)
    else:
        windows_per_series = np.full((num_series), 1)
    # print("windows pre: ", windows_per_series.shape)
    if train: 
        windows_per_series -= (data_start+stride_size-1) // stride_size
    # print("data_start: ", data_start.shape)
    # print(data_start)
    # print("windows: ", windows_per_series.shape)
    # print(windows_per_series)
    total_windows = np.sum(windows_per_series)
    if train:
        x_input = np.zeros((total_windows, input_size, 1 + num_covariates), dtype='float32')
        label = np.zeros((total_windows, input_size), dtype='float32')
        v_input = np.zeros((total_windows, 2), dtype='float32')
    else:
        x_input = np.zeros((total_windows, input_size, 1 + num_covariates), dtype='float32')
        label = np.zeros((total_windows, input_size), dtype='float32')
        v_input = np.zeros((total_windows, 2), dtype='float32')
    # cov = 4: ground truth + day_of_week + month_of_year + quarter_of_year + year + num_series
    count = 0
    if not train:
        covariates = covariates[-time_len:]
    for series in trange(num_series):
        cov_age = stats.zscore(np.arange(total_time-data_start[series]))
        if train:
            covariates[data_start[series]:time_len, 0] = cov_age[:time_len-data_start[series]]
        else:
            covariates[:, 0] = cov_age[-time_len:]
        for i in range(windows_per_series[series]):
            if train:
                window_start = stride_size*i+data_start[series]
                window_end = window_start+input_size
            else:
                window_start = 0
                window_end = input_size
            '''
            print("x: ", x_input[count, 1:, 0].shape)
            print("window start: ", window_start)
            print("window end: ", window_end)
            print("data: ", data.shape)
            print("d: ", data[window_start:window_end-1, series].shape)
            '''
            x_input[count, :, 0] = data[window_start:window_end, series]
            x_input[count, :, 1:1+num_covariates] = covariates[window_start:window_end, :]
            label[count, :] = data[window_start+1:window_end+1, series]
            nonzero_sum = (x_input[count, :, 0]!=0).sum()
            if nonzero_sum == 0:
                v_input[count, 0] = 0
            else:
                v_input[count, 0] = np.true_divide(x_input[count, :, 0].sum(), nonzero_sum)+1
                x_input[count, :, 0] = x_input[count, :, 0]/v_input[count, 0]
                if train:
                    label[count, :] = label[count, :]/v_input[count, 0]
            count += 1
    x_input = np.transpose(x_input, (0, 2, 1))
    prefix = os.path.join(save_path, 'train_' if train else 'test_')
    np.save(prefix+'data_'+save_name, x_input)
    np.save(prefix+'v_'+save_name, v_input)
    np.save(prefix+'label_'+save_name, label)


def gen_covariates(times, name, num_covariates):
    covariates = np.zeros((times.shape[0], num_covariates))
    if name == 'traffic_processed_OT=hourly.csv':
        for i, input_time in enumerate(times):
            covariates[i, 0] = input_time.weekday()
            covariates[i, 1] = input_time.month
            covariates[i, 2] = input_time.quarter
            covariates[i, 3] = input_time.year
            covariates[i, 4] = input_time.hour
    elif name == 'traffic_processed_OT=daily.csv':
        for i, input_time in enumerate(times):
            covariates[i, 0] = input_time.weekday()
            covariates[i, 1] = input_time.month
            covariates[i, 2] = input_time.quarter
            covariates[i, 3] = input_time.year
    elif name in ['ETTm1_processed_OT=quarterly.csv', 'ETTm2_processed_OT=quarterly.csv']:
        for i, input_time in enumerate(times):
            covariates[i, 0] = input_time.weekday()
            covariates[i, 1] = input_time.month
            covariates[i, 2] = input_time.quarter
            covariates[i, 3] = input_time.year
            covariates[i, 4] = input_time.hour
            covariates[i, 5] = input_time.minute
    elif name in ['ETTm1_processed_OT=hourly.csv', 'ETTm2_processed_OT=hourly.csv']:
        for i, input_time in enumerate(times):
            covariates[i, 0] = input_time.weekday()
            covariates[i, 1] = input_time.month
            covariates[i, 2] = input_time.quarter
            covariates[i, 3] = input_time.year
            covariates[i, 4] = input_time.hour
    elif name in ['ETTm1_processed_OT=daily.csv', 'ETTm2_processed_OT=daily.csv']:
        for i, input_time in enumerate(times):
            covariates[i, 0] = input_time.weekday()
            covariates[i, 1] = input_time.month
            covariates[i, 2] = input_time.quarter
            covariates[i, 3] = input_time.year
    elif name == 'weather_processed_OT=10-minute.csv':
        for i, input_time in enumerate(times):
            covariates[i, 0] = input_time.weekday()
            covariates[i, 1] = input_time.month
            covariates[i, 2] = input_time.quarter
            covariates[i, 3] = input_time.year
            covariates[i, 4] = input_time.hour
            covariates[i, 5] = input_time.minute
    elif name == 'weather_processed_OT=hourly.csv':
        for i, input_time in enumerate(times):
            covariates[i, 0] = input_time.weekday()
            covariates[i, 1] = input_time.month
            covariates[i, 2] = input_time.quarter
            covariates[i, 3] = input_time.year
    elif name == 'weather_processed_OT=daily.csv':
        for i, input_time in enumerate(times):
            covariates[i, 0] = input_time.weekday()
            covariates[i, 1] = input_time.month
            covariates[i, 2] = input_time.quarter
    elif name == 'weather_processed_OT=weekly.csv':
        for i, input_time in enumerate(times):
            covariates[i, 0] = input_time.month
            covariates[i, 1] = input_time.quarter
    for i in range(0,num_covariates):
        covariates[:,i] = stats.zscore(covariates[:,i])
    return covariates[:, :num_covariates]


def __loss__(out, target, dic=None):
    criterion = nn.MSELoss()
    return criterion(out, target)

def evaluate(model, loss_fn, test_loader, covariates, params, future, cal_days):
    model.eval()
    with torch.no_grad():
      # Test_loader: 
      # test_batch ([batch_size, 1+cov_dim, train_window])
      # v ([batch_size, 2])
      # labels ([batch_size, train_window])
        for i, (test_batch, v, labels) in enumerate(tqdm(test_loader)):
            labels = labels.unsqueeze(1).to(params.device)
            out = model(test_batch)
            ci = 0
            output = out[:, :, out.size(2) - 1].view(out.size(0), out.size(1), 1)
            output = torch.cat((output, covariates[:, :, ci].view(covariates.size(0), covariates.size(1), 1)), 1)
            out = torch.cat((test_batch, output), dim=2)
            for i in range(future - 1):
                inp = out
                out= model(inp)
                ci += 1
                output = out[:, :, out.size(2) - 1].view(out.size(0), out.size(1), 1)
                output = torch.cat((output, covariates[:, :, ci].view(covariates.size(0), covariates.size(1), 1)), 1)
                out = torch.cat((inp, output), dim=2)
            out = out[:, 0, :].view(out.size(0), 1, out.size(2))
            v = np.repeat(v[:, 0].reshape(v.shape[0], 1, 1), repeats=pred_days, axis=2)
            out_cal_days = out[:, :, -cal_days:]
            out_cal_days = torch.mul(out_cal_days, v)
            labels_cal_days = labels[:, :, -cal_days:]
            loss = loss_fn(out_cal_days, labels_cal_days)
            labels_cal_days = labels_cal_days.view(labels_cal_days.size(0), labels_cal_days.size(2))
            out_cal_days = out_cal_days.view(out_cal_days.size(0), out_cal_days.size(2))
            torch.save(labels_cal_days, f"{abs_path}/experiments/TCN/labels_cal_days.pth")
            torch.save(out_cal_days, f"{abs_path}/experiments/TCN/out_cal_days.pth")
            print(f"Actual value: {labels_cal_days}")
            print(f"Predict value: {out_cal_days}")
    return loss


def train(model: nn.Module,
          optimizer: optim,
          loss_fn,
          train_loader: DataLoader,
          params: utils.Params) -> float:
    model.train()
    loss_epoch = np.zeros(len(train_loader))
    # Train_loader:
    # train_batch ([batch_size, 1+cov_dim, train_window])
    # labels_batch ([batch_size, train_window])
    for i, (train_batch, labels_batch) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        labels_batch = labels_batch.unsqueeze(1).to(params.device)
        out = model(train_batch)
        loss = loss_fn(out, labels_batch)

        loss.backward()
        optimizer.step()
        loss_epoch[i] = loss
    return loss_epoch


if __name__ == "__main__":
    global save_path
    
    config = {
        'traffic_processed_OT=hourly.csv': {'num_covariates': 5, 'lookback_size': 48},
        'traffic_processed_OT=daily.csv': {'num_covariates': 4, 'lookback_size': 7},
        'ETTm1_processed_OT=quarterly.csv': {'num_covariates': 6, 'lookback_size': 48},
        'ETTm1_processed_OT=hourly.csv': {'num_covariates': 5, 'lookback_size': 24},
        'ETTm1_processed_OT=daily.csv': {'num_covariates': 4, 'lookback_size': 7},
        'ETTm2_processed_OT=quarterly.csv': {'num_covariates': 6, 'lookback_size': 48},
        'ETTm2_processed_OT=hourly.csv': {'num_covariates': 5, 'lookback_size': 24},
        'ETTm2_processed_OT=daily.csv': {'num_covariates': 4, 'lookback_size': 7},
        'weather_processed_OT=10-minute.csv': {'num_covariates': 6, 'lookback_size': 24},
        'weather_processed_OT=hourly.csv': {'num_covariates': 4, 'lookback_size': 24},
        'weather_processed_OT=daily.csv': {'num_covariates': 3, 'lookback_size': 7},
        'weather_processed_OT=weekly.csv': {'num_covariates': 2, 'lookback_size': 14},
    }
    args = parser.parse_args()
    save_name = args.dataset_processed
    name = args.origin_dataset
    window_size = int(args.pred_length)+config[name]['lookback_size']
    stride_size = 7
    input_size = window_size-stride_size
    num_covariates = config[name]['num_covariates']
    pred_days = int(args.pred_length)
    
    save_path = f"{abs_path}/{args.data_folder}/{args.dataset_processed}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    raw_file_path = f"{abs_path}/../../data/{args.origin_datasets}"
    result = []
    print(f"--- Preprocess data {name}: ---")
    data_frame = pd.read_csv(f"{raw_file_path}/{name}", keep_default_na=False)
    data_frame["date"] = pd.to_datetime(data_frame["date"])
    ######
    column_list = data_frame.columns.tolist()
    column_list.remove('date')
    scaler = StandardScaler()
    data_frame[column_list] = scaler.fit_transform(data_frame[column_list])
    ######
    row_norm = list(data_frame.iloc[data_frame.shape[0]-pred_days:data_frame.shape[0], -1].values)
    data_frame.fillna(0, inplace=True)
    total_time = data_frame.shape[0]
    num_series = data_frame.shape[1]-1
    covariates = gen_covariates(data_frame["date"], name, num_covariates)
    train_data = data_frame.iloc[:-input_size, 1:].values
    test_data = data_frame.iloc[-input_size-1:, 1:].values
    data_start = (train_data!=0).argmax(axis=0) # find first nonzero value in each time series
        
    prep_data(train_data, covariates, data_start)
    prep_data(test_data, covariates, data_start, train=False)

    covariates_test = covariates[-input_size:, :]
    covariates_test = np.repeat(covariates_test.reshape(1, covariates_test.shape[1], covariates_test.shape[0]), repeats=num_series, axis=0)
    covariates_test = torch.from_numpy(covariates_test).float()

    print('--- TCN.Train ---')

    # Load the parameters from json file
    args = parser.parse_args()
    model_dir = f"{abs_path}/experiments/{args.model_name}"
    json_path = f"{model_dir}/params.json"
    data_dir = f"{abs_path}/{args.data_folder}/{args.dataset_processed}"
    assert os.path.isfile(json_path), f'No json configuration file found at {json_path}'
    params = utils.Params(json_path)
    params.cov_dim = num_covariates
    params.num_inputs = num_covariates+1

    params.model_dir = model_dir

    params.device = torch.device('cpu')
    torch.manual_seed(230)
    print('--- Using cpu... ---')
    model = TemporalConvNet(
        num_inputs=params.num_inputs,
        num_channels=params.num_channels,
        kernel_size=params.kernel_size,
        dropout=params.tcn_dropout,
        init=True,
    )

    utils.set_logger(os.path.join(model_dir, 'train.log'))
    print('--- Loading the datasets... ---')

    train_set = TrainDataset(data_dir, args.dataset_processed)
    test_set = TestDataset(data_dir, args.dataset_processed)
    sampler = WeightedSampler(data_dir, args.dataset_processed) # Use weighted sampler instead of random sampler
    train_loader = DataLoader(train_set, batch_size=params.batch_size, sampler=sampler, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=params.predict_batch, sampler=RandomSampler(test_set), num_workers=4)
    print('--- Loading complete. ---')

    print(f'--- Model: \n{str(model)} ---')
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # fetch loss function
    loss_fn = __loss__
    
    # Train the model
    if args.is_training=="True": 
        print('--- Starting training for {} epoch(s) ---'.format(params.num_epochs))
        print('--- begin training and evaluation--- ')
        best_test = float('inf')
        train_len = len(train_loader)
        loss_summary = np.zeros((train_len * params.num_epochs))
        for epoch in range(params.num_epochs):
            print('--- Epoch {}/{} ---'.format(epoch + 1, params.num_epochs))
            loss_summary[epoch * train_len:(epoch + 1) * train_len] = train(model, optimizer, loss_fn, train_loader, params)
            loss_test = evaluate(model, loss_fn, test_loader, covariates_test, params, future=input_size, cal_days=pred_days)
            is_best = loss_test <= best_test

            # Save weights
            utils.save_checkpoint({'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'optim_dict': optimizer.state_dict()},
                                epoch=epoch,
                                is_best=is_best,
                                checkpoint=params.model_dir)

            if is_best:
                print('--- Found new best loss ---')
                best_test = loss_test
                print(best_test)

            print('--- Current Best ND is: %.5f ---' % best_test)
            
        print('--- TCN.Eval ---')
        
        # Create the input data pipeline
        print('--- Loading the datasets... ---')

        test_set = TestDataset(data_dir, args.dataset_processed)
        test_loader = DataLoader(test_set, batch_size=params.predict_batch, sampler=RandomSampler(test_set), num_workers=4)
        print('--- done. ---')

        print(f'--- model: {model} ---')
        loss_fn = __loss__

        print('--- Starting evaluation ---')

        # Reload weights from the saved file
        utils.load_checkpoint(os.path.join(model_dir, args.restore_file + '.pth.tar'), model)

        loss_final = evaluate(model, loss_fn, test_loader, covariates_test, params, future=input_size, cal_days=pred_days)
        print(loss_final)
    else:
        try:
            print('--- TCN.Eval ---')
        
            # Create the input data pipeline
            print('--- Loading the datasets... ---')

            test_set = TestDataset(data_dir, args.dataset_processed)
            test_loader = DataLoader(test_set, batch_size=params.predict_batch, sampler=RandomSampler(test_set), num_workers=4)
            print('--- done. ---')

            print(f'--- model: {model} ---')
            loss_fn = __loss__

            print('--- Starting evaluation ---')

            # Reload weights from the saved file
            utils.load_checkpoint(os.path.join(f'{abs_path}/../result/models/TCN/', f'{name[:-4]}_{args.pred_length}' + '.pth.tar'), model)

            loss_final = evaluate(model, loss_fn, test_loader, covariates_test, params, future=input_size, cal_days=pred_days)
            print(loss_final)
        except:
            print('--- Starting training for {} epoch(s) ---'.format(params.num_epochs))
            print('--- begin training and evaluation--- ')
            best_test = float('inf')
            train_len = len(train_loader)
            loss_summary = np.zeros((train_len * params.num_epochs))
            for epoch in range(params.num_epochs):
                print('--- Epoch {}/{} ---'.format(epoch + 1, params.num_epochs))
                loss_summary[epoch * train_len:(epoch + 1) * train_len] = train(model, optimizer, loss_fn, train_loader, params)
                loss_test = evaluate(model, loss_fn, test_loader, covariates_test, params, future=input_size, cal_days=pred_days)
                is_best = loss_test <= best_test

                # Save weights
                utils.save_checkpoint({'epoch': epoch + 1,
                                    'state_dict': model.state_dict(),
                                    'optim_dict': optimizer.state_dict()},
                                    epoch=epoch,
                                    is_best=is_best,
                                    checkpoint=params.model_dir)

                if is_best:
                    print('--- Found new best loss ---')
                    best_test = loss_test
                    print(best_test)

                print('--- Current Best ND is: %.5f ---' % best_test)

            print('--- TCN.Eval ---')

            # Create the input data pipeline
            print('--- Loading the datasets... ---')

            test_set = TestDataset(data_dir, args.dataset_processed)
            test_loader = DataLoader(test_set, batch_size=params.predict_batch, sampler=RandomSampler(test_set), num_workers=4)
            print('--- done. ---')

            print(f'--- model: {model} ---')
            loss_fn = __loss__

            print('--- Starting evaluation ---')

            # Reload weights from the saved file
            utils.load_checkpoint(os.path.join(model_dir, args.restore_file + '.pth.tar'), model)

            loss_final = evaluate(model, loss_fn, test_loader, covariates_test, params, future=input_size, cal_days=pred_days)
            print(loss_final)
            
    print(f"Save prediction of data {name}:")
    model_path_TCN = f"{abs_path}/experiments/TCN/"
    Act_TCN_norm = torch.load(f"{model_path_TCN}/labels_cal_days.pth")
    Pred_TCN_norm = torch.load(f"{model_path_TCN}/out_cal_days.pth")
    actual_val_norm = Act_TCN_norm[findByRow(Act_TCN_norm, row=row_norm)]
    prediction_norm = Pred_TCN_norm[findByRow(Act_TCN_norm, row=row_norm)]
    target_index = np.where(Act_TCN_norm==actual_val_norm)[0][0]
    Act_TCN = torch.cat((Act_TCN_norm[0:target_index], Act_TCN_norm[target_index+1:]), dim=0)
    Act_TCN = torch.cat((Act_TCN, actual_val_norm.reshape(-1, Act_TCN_norm.size(1))), dim=0)
    Act_TCN = torch.transpose(Act_TCN, 1, 0)
    Act_TCN = np.transpose(scaler.inverse_transform(Act_TCN))
    actual_val = Act_TCN[-1]
    Pred_TCN = torch.cat((Pred_TCN_norm[0:target_index], Pred_TCN_norm[target_index+1:]), dim=0)
    Pred_TCN = torch.cat((Pred_TCN, prediction_norm.reshape(-1, Pred_TCN_norm.size(1))), dim=0)
    Pred_TCN = torch.transpose(Pred_TCN, 1, 0)
    Pred_TCN = np.transpose(scaler.inverse_transform(Pred_TCN))
    prediction = Pred_TCN[-1]
    temp_result_act_norm = list()
    temp_result_pred_norm = list()
    temp_result_act = list()
    temp_result_pred = list()
    for i in range(len(prediction_norm)):
        temp_result_act_norm.append(np.round(actual_val_norm[i].numpy(), decimals=4))
        temp_result_pred_norm.append(np.round(prediction_norm[i].numpy(), decimals=4))
        temp_result_act.append(np.round(actual_val[i], decimals=4))
        temp_result_pred.append(np.round(prediction[i], decimals=4))
    temp_result_act_norm = pd.DataFrame(temp_result_act_norm, columns=[f"Label_norm_{name[:-4]}"])
    temp_result_pred_norm = pd.DataFrame(temp_result_pred_norm, columns=[f'TCN_norm_{name[:-4]}'])
    temp_result_act = pd.DataFrame(temp_result_act, columns=[f"Label_{name[:-4]}"])
    temp_result_pred = pd.DataFrame(temp_result_pred, columns=[f'TCN_{name[:-4]}'])
    temp_result = pd.concat([temp_result_act_norm, temp_result_pred_norm, temp_result_act, temp_result_pred], axis=1)
    result.append(temp_result)
    result = pd.concat(result, axis=1)
    result.to_csv(f"{abs_path}/experiments/TCN/Result_{name}", index=False)
        
    print("finished!!!")