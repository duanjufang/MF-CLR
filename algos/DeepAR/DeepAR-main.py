import os
import pandas as pd
import numpy as np
import math
from scipy import stats
from tqdm import trange
import logging
import argparse
import torch
import torch.optim as optim
import datetime
from sklearn.preprocessing import StandardScaler

import train
import utils
import model.net as net
from dataloader import *
from evaluate import evaluate
from torch.utils.data.sampler import RandomSampler
torch.set_printoptions(threshold=np.inf)

abs_path = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--is-training', default="True", help='Whether to retrain the model')
parser.add_argument('--origin-datasets', default='PUBLIC_DATASETS', help='Parent dir of the origin datasets')
parser.add_argument('--origin-dataset', default='ETTm1_processed_OT=quarterly.csv', help='Name of the origin dataset')
parser.add_argument('--data-folder', default='data', help='Parent dir of the dataset')
parser.add_argument('--dataset-processed', default='elect', help='Name of the dataset')
parser.add_argument('--model-name', default='DeepAR', help='Directory containing params.json')
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


def prep_data(data, covariates, data_start, train=True):
    time_len = data.shape[0]
    input_size = window_size-stride_size
    windows_per_series = np.full((num_series), (time_len-input_size) // stride_size)
    if train: 
        windows_per_series -= (data_start+stride_size-1) // stride_size
    total_windows = np.sum(windows_per_series)
    x_input = np.zeros((total_windows, window_size, 1 + num_covariates + 1), dtype='float32')
    label = np.zeros((total_windows, window_size), dtype='float32')
    v_input = np.zeros((total_windows, 2), dtype='float32')
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
            else:
                window_start = stride_size*i
            window_end = window_start+window_size
            x_input[count, 1:, 0] = data[window_start:window_end-1, series]
            x_input[count, :, 1:1+num_covariates] = covariates[window_start:window_end, :]
            x_input[count, :, -1] = series
            label[count, :] = data[window_start:window_end, series]
            nonzero_sum = (x_input[count, 1:input_size, 0]!=0).sum()
            if nonzero_sum == 0:
                v_input[count, 0] = 0
            else:
                v_input[count, 0] = np.true_divide(x_input[count, 1:input_size, 0].sum(),nonzero_sum)+1
                x_input[count, :, 0] = x_input[count, :, 0]/v_input[count, 0]
                if train:
                    label[count, :] = label[count, :]/v_input[count, 0]
            count += 1
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
    num_covariates = config[name]['num_covariates']
    pred_days = int(args.pred_length)
    
    save_path = f"{abs_path}/{args.data_folder}/{args.dataset_processed}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    raw_file_path = f"{abs_path}/../../data/{args.origin_datasets}"
    result = []
    print(f"Preprocess data {name}:")
    data_frame = pd.read_csv(f"{raw_file_path}/{name}", keep_default_na=False)
    data_frame["date"] = pd.to_datetime(data_frame["date"])
    
    ######
    column_list = data_frame.columns.tolist()
    column_list.remove('date')
    scaler = StandardScaler()
    data_frame[column_list] = scaler.fit_transform(data_frame[column_list])
    ######
        
    row_norm = list(data_frame.iloc[data_frame.shape[0]-window_size:data_frame.shape[0], -1].values)
    data_frame.fillna(0, inplace=True)
    total_time = data_frame.shape[0]
    num_series = data_frame.shape[1]-1
    covariates = gen_covariates(data_frame["date"], name, num_covariates)
    train_data = data_frame.iloc[:-window_size, 1:].values
    test_data = data_frame.iloc[-window_size:, 1:].values
    data_start = (train_data!=0).argmax(axis=0) # find first nonzero value in each time series
        
    prep_data(train_data, covariates, data_start)
    prep_data(test_data, covariates, data_start, train=False)

    logger = logging.getLogger('DeepAR.Train')

    # Load the parameters from json file
    args = parser.parse_args()
    model_dir = f"{abs_path}/experiments/{args.model_name}"
    json_path = f"{model_dir}/params.json"
    data_dir = f"{abs_path}/{args.data_folder}/{args.dataset_processed}"
    assert os.path.isfile(json_path), f'No json configuration file found at {json_path}'
    params = utils.Params(json_path)
    params.train_window = window_size
    params.test_window = window_size
    params.predict_start = config[name]['lookback_size']
    params.test_predict_start = config[name]['lookback_size']
    params.predict_steps = pred_days
    params.cov_dim = num_covariates

    params.relative_metrics = args.relative_metrics
    params.sampling = args.sampling
    params.model_dir = model_dir

    params.device = torch.device('cpu')
    torch.manual_seed(230)
    logger.info('Using cpu...')
    model = net.Net(params)

    utils.set_logger(os.path.join(model_dir, 'train.log'))
    logger.info('Loading the datasets...')

    train_set = TrainDataset(data_dir, args.dataset_processed)
    test_set = TestDataset(data_dir, args.dataset_processed)
    sampler = WeightedSampler(data_dir, args.dataset_processed) # Use weighted sampler instead of random sampler
    train_loader = DataLoader(train_set, batch_size=params.batch_size, sampler=sampler, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=params.predict_batch, sampler=RandomSampler(test_set), num_workers=4)
    logger.info('Loading complete.')

    logger.info(f'Model: \n{str(model)}')
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # fetch loss function
    loss_fn = net.loss_fn
    # loss_fn = net.loss_tcn
    
    # Train the model
    if args.is_training=="True": 
        logger.info('Starting training for {} epoch(s)'.format(params.num_epochs))
        train.train_and_evaluate(model,
                        train_loader,
                        test_loader,
                        optimizer,
                        loss_fn,
                        params,
                        args.restore_file)
        
        logger = logging.getLogger('DeepAR.Eval')

        # Create the input data pipeline
        logger.info('Loading the datasets...')

        test_set = TestDataset(data_dir, args.dataset_processed)
        test_loader = DataLoader(test_set, batch_size=params.predict_batch, sampler=RandomSampler(test_set), num_workers=4)
        logger.info('- done.')

        print('model: ', model)
        loss_fn = net.loss_fn

        logger.info('Starting evaluation')

        # Reload weights from the saved file
        utils.load_checkpoint(os.path.join(model_dir, args.restore_file + '.pth.tar'), model)

        test_metrics = evaluate(model, loss_fn, test_loader, params, -1, params.sampling)
        save_path = os.path.join(model_dir, 'metrics_test_{}.json'.format(args.restore_file))
        utils.save_dict_to_json(test_metrics, save_path)

        save_path = f"{abs_path}/{args.data_folder}/{save_name}"
    
    else:
        try:
            logger = logging.getLogger('DeepAR.Eval')

            # Create the input data pipeline
            logger.info('Loading the datasets...')

            test_set = TestDataset(data_dir, args.dataset_processed)
            test_loader = DataLoader(test_set, batch_size=params.predict_batch, sampler=RandomSampler(test_set), num_workers=4)
            logger.info('- done.')

            print('model: ', model)
            loss_fn = net.loss_fn

            logger.info('Starting evaluation')

            # Reload weights from the saved file
            utils.load_checkpoint(os.path.join(f'{abs_path}/../result/models/DeepAR/', f'{name[:-4]}_{args.pred_length}' + '.pth.tar'), model)

            test_metrics = evaluate(model, loss_fn, test_loader, params, -1, params.sampling)
            save_path = os.path.join(model_dir, 'metrics_test_{}.json'.format(args.restore_file))
            utils.save_dict_to_json(test_metrics, save_path)

            save_path = f"{abs_path}/{args.data_folder}/{save_name}"
        except:
            logger.info('Starting training for {} epoch(s)'.format(params.num_epochs))
            train.train_and_evaluate(model,
                            train_loader,
                            test_loader,
                            optimizer,
                            loss_fn,
                            params,
                            args.restore_file)

            logger = logging.getLogger('DeepAR.Eval')

            # Create the input data pipeline
            logger.info('Loading the datasets...')

            test_set = TestDataset(data_dir, args.dataset_processed)
            test_loader = DataLoader(test_set, batch_size=params.predict_batch, sampler=RandomSampler(test_set), num_workers=4)
            logger.info('- done.')

            print('model: ', model)
            loss_fn = net.loss_fn

            logger.info('Starting evaluation')

            # Reload weights from the saved file
            utils.load_checkpoint(os.path.join(model_dir, args.restore_file + '.pth.tar'), model)

            test_metrics = evaluate(model, loss_fn, test_loader, params, -1, params.sampling)
            save_path = os.path.join(model_dir, 'metrics_test_{}.json'.format(args.restore_file))
            utils.save_dict_to_json(test_metrics, save_path)

            save_path = f"{abs_path}/{args.data_folder}/{save_name}"
            
    print(f"Save prediction of data {name}:")
    model_path_DeepAR = f"{abs_path}/experiments/{args.model_name}/"
    Act_DeepAR_norm = torch.load(f"{model_path_DeepAR}/labels.pth")
    Pred_DeepAR_norm = torch.load(f"{model_path_DeepAR}/sample_mu.pth")
    actual_val_norm = Act_DeepAR_norm[findByRow(Act_DeepAR_norm, row=row_norm)]
    prediction_norm = Pred_DeepAR_norm[findByRow(Act_DeepAR_norm, row=row_norm)]
    target_index = np.where(Act_DeepAR_norm==actual_val_norm)[0][0]
    Act_DeepAR = torch.cat((Act_DeepAR_norm[0:target_index], Act_DeepAR_norm[target_index+1:]), dim=0)
    Act_DeepAR = torch.cat((Act_DeepAR, actual_val_norm.reshape(-1, Act_DeepAR_norm.size(1))), dim=0)
    Act_DeepAR = torch.transpose(Act_DeepAR, 1, 0)
    Act_DeepAR = np.transpose(scaler.inverse_transform(Act_DeepAR))
    actual_val = Act_DeepAR[-1]
    Pred_DeepAR = torch.cat((Pred_DeepAR_norm[0:target_index], Pred_DeepAR_norm[target_index+1:]), dim=0)
    Pred_DeepAR = torch.cat((Pred_DeepAR, prediction_norm.reshape(-1, Pred_DeepAR_norm.size(1))), dim=0)
    Pred_DeepAR = torch.transpose(Pred_DeepAR, 1, 0)
    Pred_DeepAR = np.transpose(scaler.inverse_transform(Pred_DeepAR))
    prediction = Pred_DeepAR[-1]
    temp_result_act_norm = list()
    temp_result_pred_norm = list()
    temp_result_act = list()
    temp_result_pred = list()
    for i in range(len(prediction_norm)):
        temp_result_act_norm.append(np.round(actual_val_norm[24+i].numpy(), decimals=4))
        temp_result_pred_norm.append(np.round(prediction_norm[i].numpy(), decimals=4))
        temp_result_act.append(np.round(actual_val[24+i], decimals=4))
        temp_result_pred.append(np.round(prediction[i], decimals=4))
    temp_result_act_norm = pd.DataFrame(temp_result_act_norm, columns=[f"Label_norm_{name[:-4]}"])
    temp_result_pred_norm = pd.DataFrame(temp_result_pred_norm, columns=[f'DeepAR_norm_{name[:-4]}'])
    temp_result_act = pd.DataFrame(temp_result_act, columns=[f"Label_{name[:-4]}"])
    temp_result_pred = pd.DataFrame(temp_result_pred, columns=[f'DeepAR_{name[:-4]}'])
    temp_result = pd.concat([temp_result_act_norm, temp_result_pred_norm, temp_result_act, temp_result_pred], axis=1)
    result.append(temp_result)
    result = pd.concat(result, axis=1)
    result.to_csv(f"{abs_path}/experiments/DeepAR/Result_{name}.csv", index=False)
            
    print("finished!!!")