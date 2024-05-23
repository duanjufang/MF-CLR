import os
import uuid
import time
import torch
import warnings
import argparse
import numpy as np
import pandas as pd
from MFCLR import MF_CLR
from data.dataset_preprocessing_forecast import ETT_processing, traffic_processing, weather_processing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from algos.ARIMA import AutoARIMA
from algos.lightGBM import fit_lightGBM, lightGBM_predict
from algos.informer.interface import train_and_predict
from algos.TLoss.interface import Unsupervided_Scalable as TLoss
from algos.TS_TCC.interface import TS_TCC
from algos.TNC.interface import TNC
from algos.TS2Vec.ts2vec import TS2Vec
from algos.contrastive_predictive_coding.interface import Contrasive as CPC
from algos.CosT.cost import CoST
from algos.TFC.code.TFC.interface import TFC_inter



warnings.filterwarnings('ignore')
abs_path = os.path.dirname(os.path.realpath(__file__))


class Public_Dataset_MFCLR():  
    def __init__(self, args):
        assert args.dataset in ["traffic", "ETTm1", "ETTm2", "weather"]
        assert args.ds in ["traffic", "ETT", "weather"]
        self.origin_file = args.dataset
        self.test_file = args.dataset + "_processed_OT=" + args.ot_granu
        self.data = args.ds
        self.fcst_granu = args.ot_granu
        self.enc_len = args.enc_len
        self.window_len = args.lookback
        self.pred_len = args.pred_len
        self.epoch = args.epochs
        self.training = args.is_training
        self.saving = args.save
        self.learning_rate = args.lr
        self.projector_dim = args.ph_dim
        self.batch_size = args.batch_size
        if self.saving is True :
            os.makedirs("results/models/MFCLR/", exist_ok=True)
        if args.use_gpu is False:
            self.device = 'cpu'
        else:
            self.device = 'cuda'

        self.output_dims = args.out_dim
        self.hidden_dims = args.hidden_dim
        self.depth = args.depth
        if args.ds == "traffic" :
            assert args.ot_granu in ["hourly", "daily"]
            if args.ot_granu == "hourly" :
                self.y_dim = 0
                self.pred_len_ori = 1 * self.pred_len
                self.grain_split_list = [401, 862]
            elif args.ot_granu == "daily" :
                self.y_dim = 400
                self.pred_len_ori = 24 * self.pred_len
                self.grain_split_list = [400, 862]
            self.total_dim = 862
        elif args.ds == "ETT" :
            assert args.ot_granu in ["quarterly", "hourly", "daily"]
            if args.ot_granu == "quarterly" :
                self.y_dim = 0
                self.pred_len_ori = 1 * self.pred_len
                self.grain_split_list = [3, 5, 7]
            elif args.ot_granu == "hourly" :
                self.y_dim = 2
                self.pred_len_ori = 4 * self.pred_len
                self.grain_split_list = [2, 5, 7]
            elif args.ot_granu == "daily" :
                self.y_dim = 4
                self.pred_len_ori = 96 * self.pred_len
                self.grain_split_list = [2, 4, 7]
            self.total_dim = 7
        elif args.ds == "weather" :
            assert args.ot_granu in ["10-minute", "hourly", "daily", "weekly"]
            if args.ot_granu == "10-minute" :
                self.y_dim = 0
                self.pred_len_ori = 1 * self.pred_len
                self.grain_split_list = [5, 12, 15, 21]
            elif args.ot_granu == "hourly" :
                self.y_dim = 4
                self.pred_len_ori = 6 * self.pred_len
                self.grain_split_list = [4, 12, 15, 21]
            elif args.ot_granu == "daily" :
                self.y_dim = 11
                self.pred_len_ori = 144 * self.pred_len
                self.grain_split_list = [4, 11, 15, 21]
            elif args.ot_granu == "weekly" :
                self.y_dim = 14
                self.pred_len_ori = 1008 * self.pred_len
                self.grain_split_list = [4, 12, 14, 21]
            self.total_dim = 21
        
    
    def read_file(self):
        return pd.read_csv(f"{abs_path}" + f"/data/{self.data}/" + self.origin_file + ".csv")
    

    def gen_enc_array(self, keep_date= False):
        df = self.read_file()
        if self.data == "ETT" :
            df = ETT_processing("ETTm1", self.fcst_granu, df)
        elif self.data == "traffic" :
            df = traffic_processing(self.fcst_granu, df)
        elif self.data == "weather" :
            df = weather_processing(self.fcst_granu, df)
        else : raise
        if keep_date is False :
            df.drop(columns= ["date"], inplace= True)  
        
        if self.data == "traffic" :
            data_array = df.values[ : ,  : -1]
            OT_array = df.values[ : , -1]
            if self.fcst_granu == "hourly" :
                data_array = np.insert(data_array, 0, OT_array, axis= 1)
            elif self.fcst_granu == "daily" :
                data_array = np.insert(data_array, 400, OT_array, axis= 1)

        elif self.data == "ETT" :
            data_array = df.values[ : ,  : -1]
            OT_array = df.values[ : , -1]
            if self.fcst_granu == "quarterly" :
                data_array = np.insert(data_array, 0, OT_array, axis= 1)
            elif self.fcst_granu == "hourly" :
                data_array = np.insert(data_array, 2, OT_array, axis= 1)
            elif self.fcst_granu == "daily" :
                data_array = np.insert(data_array, 4, OT_array, axis= 1)
        
        elif self.data == "weather" :
            minute_cols = ["p", "T", "Tpot", "Tdew"]
            hourly_cols = ["rh", "VPmax", "VPact", "VPdef", "sh", "H2OC", "rho"]
            daily_cols = ["wv", "max. wv", "wd"]
            weekly_cols = ["rain", "raining", "SWDR", "PAR", "max. PAR", "Tlog"]
            if self.fcst_granu == "10-minute" :
                minute_cols = ["OT"] + minute_cols
            elif self.fcst_granu == "hourly" :
                hourly_cols = ["OT"] + hourly_cols
            elif self.fcst_granu == "daily" :
                daily_cols = ["OT"] + daily_cols
            elif self.fcst_granu == "weekly" :
                daily_cols = ["OT"] + weekly_cols
            re_cols = minute_cols + hourly_cols + daily_cols + weekly_cols
            df = df.reindex(columns= re_cols)
            data_array = df.values

        return data_array
    

    def data_split(self, data_array_norm, _method= "cut"):
        assert _method in ["slide", "cut"]
        if _method == "cut" :
            cut_rows = int(np.floor(data_array_norm.shape[0] / self.enc_len)) * self.enc_len
            data_array_norm = data_array_norm[-cut_rows : , : ]
            win_num = data_array_norm.shape[0] // self.enc_len
            reconstruct_data = []
            for i in range(win_num):
                start = data_array_norm.shape[0]  - (win_num - i) * self.enc_len
                end = start + self.enc_len
                reconstruct_data.append(data_array_norm[start : end, : ])
            return np.array(reconstruct_data)
        elif _method == "slide" :
            win_num = data_array_norm.shape[0] - self.enc_len + 1
            reconstruct_data = []
            for i in range(win_num):
                start = i
                end = start + self.enc_len
                reconstruct_data.append(data_array_norm[start : end, : ])
            return np.array(reconstruct_data)
    

    def normalisation(self, inputArray, meanlist= None, stdlist= None):
        if meanlist is None and stdlist is None :
            meanlist, stdlist = [], []
            normalised_array = []
            for j in range(inputArray.shape[1]):
                normalised_array.append((inputArray[:, j] - np.mean(inputArray[:, j])) / np.std(inputArray[:, j]))
                meanlist.append(np.mean(inputArray[:, j]))
                stdlist.append(np.std(inputArray[:, j]))
            return np.array(normalised_array).T, meanlist, stdlist
        else :
            normalised_array = []
            for j in range(inputArray.shape[1]):
                normalised_array.append((inputArray[:, j] - meanlist[j]) / stdlist[j])
            return np.array(normalised_array).T, meanlist, stdlist


    def encoding(self, data_array_encode, test_array):
        print("fitting input size : ", data_array_encode.shape)
        model = MF_CLR(
            input_dims= self.grain_split_list[0], 
            grain_split= self.grain_split_list, 
            total_dim= self.total_dim,
            output_dims= self.output_dims,
            hidden_dims= self.hidden_dims,
            depth= self.depth,
            ph_dim= self.projector_dim,
            device= self.device,
            batch_size= self.batch_size,
            lr= self.learning_rate,
        )
        if self.training is True :
            loss_log = model.fit(
                train_data= data_array_encode,
                n_epochs= self.epoch,
                verbose= True,
            )
            if self.saving is True :
                os.makedirs("results/models/MFCLR/", exist_ok=True)
                model.save("results/models/MFCLR/" + self.test_file + ".pkl")
        else :
            try :
                model.load("results/models/MFCLR/" + self.test_file + ".pkl")
            except :
                loss_log = model.fit(
                    train_data= data_array_encode,
                    n_epochs= self.epoch,
                    verbose= True,
                )

        print("encoding input size : ", data_array_encode.shape)
        padding = 0
        train_repr = model.encode(
            data_array_encode, 
            casual= False, 
            sliding_length= None, 
            sliding_padding= padding,
        )
        print("origin train_repr.shape : ", train_repr.shape)
        train_repr = train_repr.reshape(-1, train_repr.shape[-1])
        print("train_repr.shape : ", train_repr.shape)
        test_repr = model.encode(
            test_array.reshape(1, test_array.shape[0], test_array.shape[1]), 
            casual= False, 
            sliding_length= None, 
            sliding_padding= padding,
        )
        test_repr = test_repr.reshape(-1, test_repr.shape[-1])
        print("test_repr.shape : ", test_repr.shape)
        all_repr = np.concatenate([train_repr, test_repr], axis= 0)
        print("all_repr.shape : ", all_repr.shape)
        print()
        return all_repr
    

    def dataset_construct(self, inputArray, all_repr, OT_app= True, _shuffle= True):
        
        if self.data == "traffic" :
            if self.fcst_granu == "hourly" :
                pass
            elif self.fcst_granu == "daily" :
                inputArray = inputArray[ : int(inputArray.shape[0] / 24) * 24]
                all_repr = all_repr[ : int(all_repr.shape[0] / 24) * 24]

        elif self.data == "ETT" :
            if self.fcst_granu == "quarterly" :
                pass
            elif self.fcst_granu == "hourly" :
                inputArray = inputArray[ : int(inputArray.shape[0] / 4) * 4]
                all_repr = all_repr[ : int(all_repr.shape[0] / 4) * 4]
            elif self.fcst_granu == "daily" :
                inputArray = inputArray[ : int(inputArray.shape[0] / 96) * 96]
                all_repr = all_repr[ : int(all_repr.shape[0] / 96) * 96]

        elif self.data == "weather" :
            if self.fcst_granu == "10-minute" :
                pass
            elif self.fcst_granu == "hourly" :
                inputArray = inputArray[ : int(inputArray.shape[0] / 6) * 6]
                all_repr = all_repr[ : int(all_repr.shape[0] / 6) * 6]
            elif self.fcst_granu == "daily" :
                inputArray = inputArray[ : int(inputArray.shape[0] / 144) * 144]
                all_repr = all_repr[ : int(all_repr.shape[0] / 144) * 144]
            elif self.fcst_granu == "weekly" :
                inputArray = inputArray[ : int(inputArray.shape[0] / 1008) * 1008]
                all_repr = all_repr[ : int(all_repr.shape[0] / 1008) * 1008]

        min_length = min(inputArray.shape[0], all_repr.shape[0])
        inputArray = inputArray[-min_length : ]
        all_repr = all_repr[-min_length : ]
        print("truncated size : ", inputArray.shape, all_repr.shape)
        _T = inputArray.shape[0]

        win_num = (_T - self.pred_len_ori) // self.window_len
        print(win_num, _T, self.pred_len_ori, self.window_len)
        train_x, train_y, test_x, test_y = [], [], [], []
        for i in range(win_num) :
            x_start = (_T - self.pred_len_ori) - (win_num - i) * self.window_len
            x_end = x_start + self.window_len
            y_start = x_end
            y_end = y_start + self.pred_len_ori
            if i != win_num - 1 :
                if OT_app is True :
                    this_x = all_repr[x_start : x_end, : ]
                    this_OT = inputArray[x_start : x_end, self.y_dim].reshape(-1, 1)
                    this_x = np.concatenate((this_x, this_OT), axis= 1)
                    train_x.append(this_x)
                else :
                    train_x.append(all_repr[x_start : x_end, : ])
                train_y.append([inputArray[k, self.y_dim] for k in range(y_start, y_end, self.pred_len_ori // self.pred_len)])
            else :
                if OT_app is True :
                    this_x = all_repr[x_start : x_end, : ]
                    this_OT = inputArray[x_start : x_end, self.y_dim].reshape(-1, 1)
                    this_x = np.concatenate((this_x, this_OT), axis= 1)
                    test_x.append(this_x)
                else :
                    test_x.append(all_repr[x_start : x_end, : ])
                print(y_start, self.pred_len_ori, y_end, self.pred_len, inputArray.shape)
                test_y.append([inputArray[k, self.y_dim] for k in range(y_start, y_end, self.pred_len_ori // self.pred_len)])
        
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        test_x = np.array(test_x)
        test_y = np.array(test_y)
        print("window number : ", win_num)
        print("original input size : ", inputArray.shape)
        print("train_x shape : ", train_x.shape)
        print("train_y shape : ", train_y.shape)
        print("test_x shape : ", test_x.shape)
        print("test_y shape : ", test_y.shape)

        if _shuffle is True :
            random_idx = np.random.permutation(train_x.shape[0])
            train_x = train_x[random_idx]
            train_y = train_y[random_idx]
        
        return train_x, train_y, test_x, test_y


    def fit_ridge(self, train_features, train_y, valid_features, valid_y, MAX_SAMPLES=100000):
        # If the training set is too large, subsample MAX_SAMPLES examples
        if train_features.shape[0] > MAX_SAMPLES:
            split = train_test_split(
                train_features, train_y,
                train_size= MAX_SAMPLES, random_state= 0
            )
            train_features = split[0]
            train_y = split[2]
        if valid_features.shape[0] > MAX_SAMPLES:
            split = train_test_split(
                valid_features, valid_y,
                train_size= MAX_SAMPLES, random_state= 0
            )
            valid_features = split[0]
            valid_y = split[2]
        
        alphas = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        valid_results = []
        for alpha in alphas:
            lr = Ridge(alpha= alpha).fit(train_features, train_y)
            valid_pred = lr.predict(valid_features)
            score = np.sqrt(((valid_pred - valid_y) ** 2).mean()) + np.abs(valid_pred - valid_y).mean()
            valid_results.append(score)
        best_alpha = alphas[np.argmin(valid_results)]
        
        lr = Ridge(alpha=best_alpha)
        lr.fit(train_features, train_y)
        return lr


    def cal_loss(self, preds, labels):
        assert np.ndim(preds) == np.ndim(labels) == 1
        assert preds.shape[0] == labels.shape[0]
        mse = mean_squared_error(labels, preds)
        mae = abs(preds - labels).mean()
        mape = (abs(preds - labels) / labels).mean()
        return mse, mae, mape
    

    def fcst(self):
        # encoding
        data_array = self.gen_enc_array()
        data_array_train = data_array[ : -self.pred_len_ori, : ]
        data_array_test = data_array[-self.pred_len_ori : , : ]
        data_array_train_norm, mean, std = self.normalisation(data_array_train)
        data_array_test_norm, _, _ = self.normalisation(data_array_test, mean, std)
        print("data_array_train shape : ", data_array_train.shape)
        print("data_array_train_norm shape : ", data_array_train_norm.shape)
        data_array_enc = self.data_split(data_array_train_norm)
        all_repr = self.encoding(data_array_enc, data_array_test_norm)

        data_array_norm = np.concatenate([data_array_train_norm, data_array_test_norm], axis= 0)
        train_x, train_y, test_x, test_y = self.dataset_construct(data_array_norm, all_repr)
        print()

        x_train, x_valid = train_x[ : int(train_x.shape[0] * 0.8), : ], train_x[int(train_x.shape[0] * 0.8) : , : ]
        y_train, y_valid = train_y[ : int(train_y.shape[0] * 0.8), : ], train_y[int(train_y.shape[0] * 0.8) : , : ]
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_valid = x_valid.reshape(x_valid.shape[0], -1)
        print("x_train.shape : ", x_train.shape)
        print("y_train.shape : ", y_train.shape)
        print("x_valid.shape : ", x_valid.shape)
        print("y_valid.shape : ", y_valid.shape)
        predict_model = self.fit_ridge(x_train, y_train, x_valid, y_valid)

        # forecast
        t = time.time()
        test_x = test_x.reshape(test_x.shape[0], -1)
        test_pred = predict_model.predict(test_x)
        fit_time = time.time() - t
        print(test_pred)
        print(test_y)
        print(test_pred.shape, test_y.shape)
        print("fit time : ", fit_time)
        mse, mae, mape  = self.cal_loss(test_pred.reshape(-1, ), test_y.reshape(-1, ))
        print("normalised : ", mse, mae, mape)
        print("mean : ", mean[self.y_dim], "std : ", std[self.y_dim])

        return mse, mae




class Public_Dataset_TSCLR():

    def __init__(self, args):
        assert args.dataset in ["traffic", "ETTm1", "ETTm2", "weather"]
        assert args.ds in ["traffic", "ETT", "weather"]
        self.origin_file = args.dataset
        self.test_file = args.dataset + "_processed_OT=" + args.ds
        self.clr_method = args.method
        self.data = args.ds
        self.fcst_granu = args.ot_granu
        self.enc_len = args.enc_len
        self.window_len = args.lookback
        self.pred_len = args.pred_len
        self.epoch = args.epochs
        self.training = args.is_training
        self.saving = args.save
        if self.saving is True:
            model_file = self.clr_method.replace("-","")
            os.makedirs(f"results/models/{model_file}/", exist_ok=True)
        if args.ds == "traffic" :
            assert args.ot_granu in ["hourly", "daily"]
            if args.ot_granu == "hourly" :
                self.y_dim = 0
                self.pred_len_ori = 1 * self.pred_len
            elif args.ot_granu == "daily" :
                self.y_dim = 400
                self.pred_len_ori = 24 * self.pred_len
            self.total_dim = 862
        elif args.ds == "ETT" :
            assert args.ot_granu in ["quarterly", "hourly", "daily"]
            if args.ot_granu == "quarterly" :
                self.y_dim = 0
                self.pred_len_ori = 1 * self.pred_len
            elif args.ot_granu == "hourly" :
                self.y_dim = 2
                self.pred_len_ori = 4 * self.pred_len
            elif args.ot_granu == "daily" :
                self.y_dim = 4
                self.pred_len_ori = 96 * self.pred_len
            self.total_dim = 7
        elif args.ds == "weather" :
            assert args.ot_granu in ["10-minute", "hourly", "daily", "weekly"]
            if args.ot_granu == "10-minute" :
                self.y_dim = 0
                self.pred_len_ori = 1 * self.pred_len
            elif args.ot_granu == "hourly" :
                self.y_dim = 4
                self.pred_len_ori = 6 * self.pred_len
            elif args.ot_granu == "daily" :
                self.y_dim = 11
                self.pred_len_ori = 144 * self.pred_len
            elif args.ot_granu == "weekly" :
                self.y_dim = 14
                self.pred_len_ori = 1008 * self.pred_len
            self.total_dim = 21
        
    
    def read_file(self):
        return pd.read_csv(f"{abs_path}" + f"/data/{self.data}/" + self.origin_file + ".csv")
    

    def gen_enc_array(self, keep_date= False):
        df = self.read_file()
        if self.data == "ETT" :
            df = ETT_processing("ETTm1", self.fcst_granu, df)
        elif self.data == "traffic" :
            df = traffic_processing(self.fcst_granu, df)
        elif self.data == "weather" :
            df = weather_processing(self.fcst_granu, df)
        else : raise
        if keep_date is False :
            df.drop(columns= ["date"], inplace= True)  
        
        if self.data == "traffic" :
            data_array = df.values[ : ,  : -1]
            OT_array = df.values[ : , -1]
            if self.fcst_granu == "hourly" :
                data_array = np.insert(data_array, 0, OT_array, axis= 1)
            elif self.fcst_granu == "daily" :
                data_array = np.insert(data_array, 400, OT_array, axis= 1)
        
        elif self.data == "ETT" :
            data_array = df.values[ : ,  : -1]
            OT_array = df.values[ : , -1]
            if self.fcst_granu == "quarterly" :
                data_array = np.insert(data_array, 0, OT_array, axis= 1)
            elif self.fcst_granu == "hourly" :
                data_array = np.insert(data_array, 2, OT_array, axis= 1)
            elif self.fcst_granu == "daily" :
                data_array = np.insert(data_array, 4, OT_array, axis= 1)
        
        elif self.data == "weather" :
            minute_cols = ["p", "T", "Tpot", "Tdew"]
            hourly_cols = ["rh", "VPmax", "VPact", "VPdef", "sh", "H2OC", "rho"]
            daily_cols = ["wv", "max. wv", "wd"]
            weekly_cols = ["rain", "raining", "SWDR", "PAR", "max. PAR", "Tlog"]
            if self.fcst_granu == "10-minute" :
                minute_cols = ["OT"] + minute_cols
            elif self.fcst_granu == "hourly" :
                hourly_cols = ["OT"] + hourly_cols
            elif self.fcst_granu == "daily" :
                daily_cols = ["OT"] + daily_cols
            elif self.fcst_granu == "weekly" :
                daily_cols = ["OT"] + weekly_cols
            re_cols = minute_cols + hourly_cols + daily_cols + weekly_cols
            df = df.reindex(columns= re_cols)
            data_array = df.values

        return data_array
    


    def data_split(self, data_array_norm, _method= "slide"):
        assert _method in ["slide", "cut"]
        if _method == "cut" :
            cut_rows = int(np.floor(data_array_norm.shape[0] / self.enc_len)) * self.enc_len
            data_array_norm = data_array_norm[-cut_rows : , : ]
            win_num = data_array_norm.shape[0] // self.enc_len
            reconstruct_data = []
            for i in range(win_num):
                start = data_array_norm.shape[0]  - (win_num - i) * self.enc_len
                end = start + self.enc_len
                reconstruct_data.append(data_array_norm[start : end, : ])
            return np.array(reconstruct_data)
        elif _method == "slide" :
            win_num = data_array_norm.shape[0] - self.enc_len + 1
            reconstruct_data = []
            for i in range(win_num):
                start = i
                end = start + self.enc_len
                reconstruct_data.append(data_array_norm[start : end, : ])
            return np.array(reconstruct_data)
    

    def normalisation(self, inputArray, meanlist= None, stdlist= None):
        if meanlist is None and stdlist is None :
            meanlist, stdlist = [], []
            normalised_array = []
            for j in range(inputArray.shape[1]):
                normalised_array.append((inputArray[:, j] - np.mean(inputArray[:, j])) / np.std(inputArray[:, j]))
                meanlist.append(np.mean(inputArray[:, j]))
                stdlist.append(np.std(inputArray[:, j]))
            return np.array(normalised_array).T, meanlist, stdlist
        else :
            normalised_array = []
            for j in range(inputArray.shape[1]):
                normalised_array.append((inputArray[:, j] - meanlist[j]) / stdlist[j])
            return np.array(normalised_array).T, meanlist, stdlist


    def TLoss_encoding(self, data_array_encode, test_array):
        """
        parameters meaning in  Unsupervised_Scalable:
        :params input_dims  in_channel in casula-cnn
        :params channels  channels number of channels manipulated in the casual CNN
        :params output_dims  number of output channels
        eg. input_shape(batch_size,timestep,feature)->(batch_size,feature)
        """
        print("encoding input size : ", data_array_encode.shape)
        model = TLoss(
             input_dims=data_array_encode.shape[-1],
             lr=0.001,
             batch_size=16,
             output_dims=320,
             channels = 40,
             depth=10,
             kernel_size=3,
             nb_steps= self.epoch
        )
        model.fit(data_array_encode)

        if self.training is True :
            loss_log = model.fit(data_array_encode,)
            if self.saving is True :
                model.save("results/models/TLoss/" + self.test_file + ".pkl")
        else :
            try :
                model.load("results/models/TLoss/" + self.test_file + ".pkl")

            except :
                loss_log = model.fit(data_array_encode)

        train_repr = model.encode(data_array_encode)
        print("original train_repr.shape : ", train_repr.shape)
        train_repr = train_repr.reshape(-1, train_repr.shape[-1])
        print("train_repr.shape : ", train_repr.shape)
        test_repr = model.encode(self.data_split(test_array))
        test_repr = test_repr.reshape(-1, test_repr.shape[-1])
        print("test_repr.shape : ", test_repr.shape)
        all_repr = np.concatenate([train_repr, test_repr], axis= 0)
        print("all_repr.shape : ", all_repr.shape)
        print()
        return all_repr
    

    def TSTCC_encoding(self, data_array_encode, test_array):
              
        print("encoding input size : ", data_array_encode.shape)
        model = TS_TCC(
            input_dims=data_array_encode.shape[-1],
            kernel_size=1, 
            stride = 1,
            final_out_channels = 320,
            tc_timestep=4,
            device="cpu"
        )
        if self.training is True :
            loss_log = model.fit(
                data_array_encode,
                n_epochs= self.epoch,
            )
            if self.saving is True :
                model.save_model("results/models/TSTCC/" + self.test_file + ".pkl")
        else :
            try :
                model.load_model("results/models/TSTCC/" + self.test_file + ".pkl")
            except :
                loss_log = model.fit(
                    data_array_encode,
                    n_epochs= self.epoch,
                )
            
        train_repr = model.encode(data_array_encode)
        print("original train_repr.shape : ", train_repr.shape)
        train_repr = train_repr.reshape(-1, train_repr.shape[-1])
        print("train_repr.shape : ", train_repr.shape)
        test_repr = model.encode(self.data_split(test_array, "cut"))
        test_repr = test_repr.reshape(-1, test_repr.shape[-1])
        print("test_repr.shape : ", test_repr.shape)
        all_repr = np.concatenate([train_repr, test_repr], axis= 0)
        print("all_repr.shape : ", all_repr.shape)
        print()
        return all_repr
    

    def TNC_encoding(self, data_array_encode, test_array): 
        print("encoding input size : ", data_array_encode.shape)
        model = TNC(
            encoding_size= self.enc_len,
            device='cpu',
            batch_size= 16,
            lr= 0.001,
            cv = 1,
            w = 20,
            decay= 0.005,
            mc_sample_size=20,
            augmentation= 1,
            channels = data_array_encode.shape[2],
        )
        if self.training is True :
            loss, accuracy = model.fit(data_array_encode, self.epoch)
            if self.saving is True :
                model.save("results/models/TNC/" + self.test_file + ".pkl")
        else :
            try :
                model.load("results/models/TNC/" + self.test_file + ".pkl")
            except :
                loss, accuracy = model.fit(data_array_encode, self.epoch)
        
        train_repr = model.encode(data_array_encode)
        print("original train_repr.shape : ", train_repr.shape)
        train_repr = train_repr.reshape(-1, train_repr.shape[-1])
        print("train_repr.shape : ", train_repr.shape)
        test_repr = model.encode(self.data_split(test_array))
        test_repr = test_repr.reshape(-1, test_repr.shape[-1])
        print("test_repr.shape : ", test_repr.shape)
        all_repr = np.concatenate([train_repr, test_repr], axis= 0)
        print("all_repr.shape : ", all_repr.shape)
        print()
        return all_repr


    def TS2Vec_encoding(self, data_array_encode, test_array):
        print("encoding input size : ", data_array_encode.shape)
        model = TS2Vec(
            input_dims= data_array_encode.shape[-1], 
            device= "cpu",
        )
        if self.training is True :
            loss_log = model.fit(
                train_data= data_array_encode,
                n_epochs= self.epoch,
                verbose= True
            )
            
            if self.saving is True :
                model.save("results/models/TS2Vec/" + self.test_file + ".pkl")
        else :
            try :
                model.load("results/models/TS2Vec/" + self.test_file + ".pkl")
            except :
                loss_log = model.fit(
                    train_data= data_array_encode,
                    n_epochs= self.epoch,
                    verbose= True
                )

        train_repr = model.encode(data_array_encode)
        print("original train_repr.shape : ", train_repr.shape)
        train_repr = train_repr.reshape(-1, train_repr.shape[-1])
        print("train_repr.shape : ", train_repr.shape)
        test_repr = model.encode(self.data_split(test_array, "cut"))
        test_repr = test_repr.reshape(-1, test_repr.shape[-1])
        print("test_repr.shape : ", test_repr.shape)
        all_repr = np.concatenate([train_repr, test_repr], axis= 0)
        print("all_repr.shape : ", all_repr.shape)
        print()
        return all_repr
    

    def CPC_encoding(self, data_array_encode, test_array):
        print("encoding input size : ", data_array_encode.shape)
        model = CPC(
            strides = [2, 2],
            filter_sizes = [2, 2],
            padding = [2, 1],   
            genc_input = data_array_encode.shape[-1],
            genc_hidden = 512,
            gar_hidden = 256, 
            lr = 2.0e-4,
            batch_size = 16,
            device='cpu'       
        )
        if self.training is True :
            n_epoch = self.epoch
            model.fit(data_array_encode, n_epoch)
            if self.saving is True :
                model.save_model(fn= "results/models/CPC/" + self.test_file + ".pkl")
        else :
            try :
                model.load_model("results/models/CPC/" + self.test_file + ".pkl")
            except :
                n_epoch = self.epoch
                model.fit(data_array_encode, n_epoch)

        train_repr = model.encode(data_array_encode)
        print("original train_repr.shape : ", train_repr.shape)
        train_repr = train_repr.reshape(-1, train_repr.shape[-1])
        print("train_repr.shape : ", train_repr.shape)
        test_repr = model.encode(self.data_split(test_array, "cut"))
        test_repr = test_repr.reshape(-1, test_repr.shape[-1])
        print("test_repr.shape : ", test_repr.shape)
        all_repr = np.concatenate([train_repr, test_repr], axis= 0)
        print("all_repr.shape : ", all_repr.shape)
        print()
        return all_repr
    

    def CoST_encoding(self, data_array_encode, test_array):
        """
        value is default in model code
        """
        print("encoding input size : ", data_array_encode.shape)
        model = CoST(
            input_dims=data_array_encode.shape[-1],
            kernels=[1, 2, 4, 8, 16, 32, 64, 128],
            alpha=0.0005,
            max_train_length=self.enc_len,
            device="cpu")
        if self.training is True :
            loss_log = model.fit(
                data_array_encode,
                n_epochs= self.epoch,
                verbose=True
            )
            if self.saving is True :
                model.save("results/models/CoST/" + self.test_file + ".pkl")
        else :
            try :
                model.load("results/models/CoST/" + self.test_file + ".pkl")
            except :
                loss_log = model.fit(
                    data_array_encode,
                    n_epochs= self.epoch,
                    verbose=True
                )

        train_repr = model.encode(data_array_encode)
        print("original train_repr.shape : ", train_repr.shape)
        train_repr = train_repr.reshape(-1, train_repr.shape[-1])
        print("train_repr.shape : ", train_repr.shape)
        test_repr = model.encode(self.data_split(test_array, "cut"))
        test_repr = test_repr.reshape(-1, test_repr.shape[-1])
        print("test_repr.shape : ", test_repr.shape)
        all_repr = np.concatenate([train_repr, test_repr], axis= 0)
        print("all_repr.shape : ", all_repr.shape)
        print()
        return all_repr
    

    def TFC_encoding(self, data_array_encode, test_array):
        """
        output (samples, encoding_size)
        """    
        print("encoding input size : ", data_array_encode.shape)
        model = TFC_inter(
                 device = 'cpu',
                 lr = 3e-4,
                 beta1 = 0.9,
                 beta2 = 0.99,
                 input_channels = 1,
                 kernel_size = 8,
                 stride = 1,
                 final_out_channels = 128,
                 features_len = 18,
                 dropout = 0.35,
                 batch_size = 128,
                 TSlength_aligned=data_array_encode.shape[1],
        )
        if self.training is True :
            model.fit(data_array_encode, self.epoch)
            if self.saving is True :
                model.save("results/models/TFC/" + self.test_file + ".pkl")
        else :
            try :
                model.load("results/models/TFC/" + self.test_file + ".pkl")
            except :
                model.fit(data_array_encode, self.epoch)

        train_repr = model.encode(data_array_encode)
        print("original train_repr.shape : ", train_repr.shape)
        train_repr = train_repr.reshape(-1, train_repr.shape[-1])
        print("train_repr.shape : ", train_repr.shape)
        test_array_encode = self.data_split(test_array)
        print("test_array_encode.shape : ", test_array_encode.shape)
        test_repr = model.encode(test_array_encode)
        test_repr = test_repr.reshape(-1, test_repr.shape[-1])
        print("test_repr.shape : ", test_repr.shape)
        all_repr = np.concatenate([train_repr, test_repr], axis= 0)
        print("all_repr.shape : ", all_repr.shape)
        print()
        return all_repr
    

    def dataset_construct(self, inputArray, all_repr, OT_app= True, _shuffle= True):
        if self.data == "traffic" :
            if self.fcst_granu == "hourly" :
                pass
            elif self.fcst_granu == "daily" :
                inputArray = inputArray[ : int(inputArray.shape[0] / 24) * 24]
                all_repr = all_repr[ : int(all_repr.shape[0] / 24) * 24]

        elif self.data == "ETT" :
            if self.fcst_granu == "quarterly" :
                pass
            elif self.fcst_granu == "hourly" :
                inputArray = inputArray[ : int(inputArray.shape[0] / 4) * 4]
                all_repr = all_repr[ : int(all_repr.shape[0] / 4) * 4]
            elif self.fcst_granu == "daily" :
                inputArray = inputArray[ : int(inputArray.shape[0] / 96) * 96]
                all_repr = all_repr[ : int(all_repr.shape[0] / 96) * 96]

        elif self.data == "weather" :
            if self.fcst_granu == "10-minute" :
                pass
            elif self.fcst_granu == "hourly" :
                inputArray = inputArray[ : int(inputArray.shape[0] / 6) * 6]
                all_repr = all_repr[ : int(all_repr.shape[0] / 6) * 6]
            elif self.fcst_granu == "daily" :
                inputArray = inputArray[ : int(inputArray.shape[0] / 144) * 144]
                all_repr = all_repr[ : int(all_repr.shape[0] / 144) * 144]
            elif self.fcst_granu == "weekly" :
                inputArray = inputArray[ : int(inputArray.shape[0] / 1008) * 1008]
                all_repr = all_repr[ : int(all_repr.shape[0] / 1008) * 1008]

        print("original size : ", inputArray.shape, all_repr.shape)
        min_length = min(inputArray.shape[0], all_repr.shape[0])
        inputArray = inputArray[-min_length : ]
        all_repr = all_repr[-min_length : ]
        print("truncated size : ", inputArray.shape, all_repr.shape)
        _T = inputArray.shape[0]

        win_num = (_T - self.pred_len_ori) // self.window_len
        print(win_num, _T, self.pred_len_ori, self.window_len)
        train_x, train_y, test_x, test_y = [], [], [], []
        for i in range(win_num) :
            x_start = (_T - self.pred_len_ori) - (win_num - i) * self.window_len
            x_end = x_start + self.window_len
            y_start = x_end
            y_end = y_start + self.pred_len_ori
            if i != win_num - 1 :
                if OT_app is True :
                    this_x = all_repr[x_start : x_end, : ]
                    this_OT = inputArray[x_start : x_end, self.y_dim].reshape(-1, 1)
                    this_x = np.concatenate((this_x, this_OT), axis= 1)
                    train_x.append(this_x)
                else :
                    train_x.append(all_repr[x_start : x_end, : ])
                train_y.append([inputArray[k, self.y_dim] for k in range(y_start, y_end, self.pred_len_ori // self.pred_len)])
            else :
                if OT_app is True :
                    this_x = all_repr[x_start : x_end, : ]
                    this_OT = inputArray[x_start : x_end, self.y_dim].reshape(-1, 1)
                    this_x = np.concatenate((this_x, this_OT), axis= 1)
                    test_x.append(this_x)
                else :
                    test_x.append(all_repr[x_start : x_end, : ])
                test_y.append([inputArray[k, self.y_dim] for k in range(y_start, y_end, self.pred_len_ori // self.pred_len)])
        
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        test_x = np.array(test_x)
        test_y = np.array(test_y)
        print("window number : ", win_num)
        print("pred_len_ori : ", self.pred_len_ori)
        print("original input size : ", inputArray.shape)
        print("train_x shape : ", train_x.shape)
        print("train_y shape : ", train_y.shape)
        print("test_x shape : ", test_x.shape)
        print("test_y shape : ", test_y.shape)

        if _shuffle is True :
            random_idx = np.random.permutation(train_x.shape[0])
            train_x = train_x[random_idx]
            train_y = train_y[random_idx]
        
        return train_x, train_y, test_x, test_y


    def fit_ridge(self, train_features, train_y, valid_features, valid_y, MAX_SAMPLES=100000):
        # If the training set is too large, subsample MAX_SAMPLES examples
        if train_features.shape[0] > MAX_SAMPLES:
            split = train_test_split(
                train_features, train_y,
                train_size= MAX_SAMPLES, random_state= 0
            )
            train_features = split[0]
            train_y = split[2]
        if valid_features.shape[0] > MAX_SAMPLES:
            split = train_test_split(
                valid_features, valid_y,
                train_size= MAX_SAMPLES, random_state= 0
            )
            valid_features = split[0]
            valid_y = split[2]
        
        alphas = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        valid_results = []
        for alpha in alphas:
            lr = Ridge(alpha= alpha).fit(train_features, train_y)
            valid_pred = lr.predict(valid_features)
            score = np.sqrt(((valid_pred - valid_y) ** 2).mean()) + np.abs(valid_pred - valid_y).mean()
            valid_results.append(score)
        best_alpha = alphas[np.argmin(valid_results)]
        
        lr = Ridge(alpha=best_alpha)
        lr.fit(train_features, train_y)
        return lr


    def cal_loss(self, preds, labels):
        assert np.ndim(preds) == np.ndim(labels) == 1
        assert preds.shape[0] == labels.shape[0]
        mse = mean_squared_error(labels, preds)
        mae = abs(preds - labels).mean()
        mape = (abs(preds - labels) / labels).mean()
        return mse, mae, mape

    
    def fcst(self):
        # encoding
        data_array = self.gen_enc_array()
        data_array_train = data_array[ : -self.pred_len_ori, : ]
        data_array_test = data_array[-self.pred_len_ori : , : ]
        data_array_train_norm, mean, std = self.normalisation(data_array_train)
        data_array_test_norm, _, _ = self.normalisation(data_array_test, mean, std)
        print("data_array_train shape : ", data_array_train.shape)
        print("data_array_train_norm shape : ", data_array_train_norm.shape)
        print("data_array_test shape : ", data_array_test.shape)
        print("data_array_test_norm shape : ", data_array_test_norm.shape)
        
        if self.clr_method == "T-Loss" :
            data_array_enc = self.data_split(data_array_train_norm)
            all_repr = self.TLoss_encoding(data_array_enc, data_array_test_norm)
        elif self.clr_method == "TS-TCC" :
            data_array_enc = self.data_split(data_array_train_norm, "cut")
            all_repr = self.TSTCC_encoding(data_array_enc, data_array_test_norm)
        elif self.clr_method == "TNC" :
            data_array_enc = self.data_split(data_array_train_norm)
            all_repr = self.TNC_encoding(data_array_enc, data_array_test_norm)
        elif self.clr_method == "TS2Vec" :
            data_array_enc = self.data_split(data_array_train_norm, "cut")
            all_repr = self.TS2Vec_encoding(data_array_enc, data_array_test_norm)
        elif self.clr_method == "CPC" :
            data_array_enc = self.data_split(data_array_train_norm, "cut")
            all_repr = self.CPC_encoding(data_array_enc, data_array_test_norm)
        elif self.clr_method == "CoST" :
            data_array_enc = self.data_split(data_array_train_norm, "cut")
            all_repr = self.CoST_encoding(data_array_enc, data_array_test_norm)
        elif self.clr_method == "TF-C" :
            data_array_enc = self.data_split(data_array_train_norm)
            all_repr = self.TFC_encoding(data_array_enc, data_array_test_norm)
        else : raise

        data_array_norm = np.concatenate([data_array_train_norm, data_array_test_norm], axis= 0)
        train_x, train_y, test_x, test_y = self.dataset_construct(data_array_norm, all_repr)

        x_train, x_valid = train_x[ : int(train_x.shape[0] * 0.8), : ], train_x[int(train_x.shape[0] * 0.8) : , : ]
        y_train, y_valid = train_y[ : int(train_y.shape[0] * 0.8), : ], train_y[int(train_y.shape[0] * 0.8) : , : ]
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_valid = x_valid.reshape(x_valid.shape[0], -1)
        print("x_train.shape : ", x_train.shape)
        print("y_train.shape : ", y_train.shape)
        print("x_valid.shape : ", x_valid.shape)
        print("y_valid.shape : ", y_valid.shape)
       
        # fit
        predict_model = self.fit_ridge(x_train, y_train, x_valid, y_valid)

        # forecast
        t = time.time()
        test_x = test_x.reshape(test_x.shape[0], -1)
        test_pred = predict_model.predict(test_x)
        fit_time = time.time() - t
        print(test_pred)
        print(test_y)
        print(test_pred.shape, test_y.shape)
        print("fit time : ", fit_time)
        mse, mae, mape = self.cal_loss(test_pred.reshape(-1, ), test_y.reshape(-1, ))
        print("normalised : ", mse, mae, mape)
        print("mean : ", mean[self.y_dim], "std : ", std[self.y_dim])

        return mse, mae

        


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_training', type=bool, required=True, default=1, help='status')
    parser.add_argument('--dataset', type=str, required=True, help='The dataset name, This can be set to ETTm1, ETTm2, traffic, weather')
    parser.add_argument('--method', type=str, required=True, default='mf_clr', help='model_name, options:[MF-CLR, T-Loss, TS-TCC, TNC, TS2Vec, CPC, CoST]')
    parser.add_argument('--use_gpu', type=bool, default=False, help='use_gpu')
    parser.add_argument('--batch_size', type=int, default=32, help='The batch size(defaults to 16)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate(defaluts to 0.001) for ')
    parser.add_argument('--epochs', type=int, default=30, help='The number of epochs(defaults to 30)')
    parser.add_argument('--enc_len', type=int, default=30, help='input sequence length')
    parser.add_argument('--ot_granu', type=str, default='quarterly', help='frequency of the forecast target')
    parser.add_argument('--save', type=bool, default=True, help='save the checkpoint')
    parser.add_argument('--ph_dim', type=int, default=32, help='dimension of projector')
    parser.add_argument('--out_dim', type=int, default=64, help='dimension of output')
    parser.add_argument('--hidden_dim', type=int, default=128, help='dimension of hidden')
    parser.add_argument('--depth', type=int, default=4, help='numbers of dilated convolutional layers')


    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    print('Args in experiment:')
    print(args)

    data_parser = {
        'ETTm1':{'quarterly': {'predlen':[48, 672], 'lookback':[128, 1024]}, 'hourly':{'predlen':[24, 336], 'lookback':[64, 512]}, 
                 'daily':{'predlen':[7, 30], 'lookback':[16, 64]}},
        'ETTm2':{'quarterly': {'predlen':[48, 672], 'lookback':[128, 1024]}, 'hourly':{'predlen':[24, 336], 'lookback':[64, 512]}, 
                 'daily':{'predlen':[7, 30], 'lookback':[16, 64]}},
        'traffic':{'hourly': {'predlen':[168, 336], 'lookback':[256, 512]}, 'daily':{'predlen':[7, 14], 'lookback':[16, 32]}},      
        'weather':{'10-minute':{'predlen' : [144, 432], 'lookback' : [256, 512]}, 'hourly' : {'predlen': [168, 720], 'lookback': [256, 1024]}, 
                   'daily': {'predlen' : [7, 30], 'lookback':[16, 64]}, 'weekly':{'predlen' : [4, 13], 'lookback':[8, 32]}}
    }
    
    if args.dataset in ["ETTm1", "ETTm2"] : 
        args.ds = "ETT"
    else :
        args.ds = args.dataset

    
    predlen = data_parser[args.dataset][args.ot_granu]['predlen']
    lookback = data_parser[args.dataset][args.ot_granu]['lookback']

    
    for jj in range(len(predlen)):
        args.pred_len = predlen[jj]
        args.lookback = lookback[jj]
        CLR_EXP = Public_Dataset_MFCLR(args) if args.method=='MF-CLR' else Public_Dataset_TSCLR(args)
        CLR_EXP.fcst()
        print(args.method, args.dataset, args.ot_granu, predlen[jj])
        print()
        print()
        print()
        print()

    