import os
import argparse
import time
import torch
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from data.dataset_preprocessing_imputation import ETT_processing, weather_processing, traffic_processing
from MFCLR import MF_CLR
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
        if args.dataset in ["ETTm1", "ETTm2"] :
            self.test_file = args.dataset + "_processed_OT=quarterly.csv"
        elif args.dataset == "traffic" :
            self.test_file = args.dataset + "_processed_OT=hourly.csv"
        elif args.dataset == "weather" :
            self.test_file = args.dataset + "_processed_OT=10-minute.csv"
        self.data = args.ds
        self.mask_ratio = args.mask_rate
        self.enc_len = args.enc_len
        self.win_len = 65
        self.epoch = args.epochs
        self.test_ratio = args.split
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
            self.y_dim = 0
            self.grain_split_list = [401, 862]
            self.total_dim = 862
        elif args.ds == "ETT" :
            self.y_dim = 0
            self.grain_split_list = [3, 5, 7]
            self.total_dim = 7
        elif args.ds == "weather" :
            self.y_dim = 0
            self.grain_split_list = [5, 12, 15, 21]
            self.total_dim = 21
        self.df, self.mask_df, self.mask_idx = self.read_file(0)
    

    def masking(self, input_array, mask_idx, mask_val= np.nan):
        assert mask_val in [0, np.nan]
        df_mask = input_array.copy()
        for r in mask_idx :
            df_mask[r, : ] = mask_val
        return df_mask
        
    
    def read_file(self, mask_val= np.nan):
        assert mask_val in [0, np.nan]
        if self.data == "traffic" :
            df_ori, mask_idx = traffic_processing("hourly", self.mask_ratio)
        elif self.data == "ETT" :
            df_ori, mask_idx = ETT_processing(self.origin_file, "quarterly", self.mask_ratio)
        elif self.data == "weather" :
            df_ori, mask_idx = weather_processing("10-minute", self.mask_ratio)
        df_mask = df_ori.copy()
        for r in mask_idx :
            df_mask.iloc[r, : ] = mask_val
        return df_ori, df_mask, mask_idx
    

    def gen_enc_array(self, keep_date= False, df= None):
        if df is None :
            df = self.mask_df.copy()
        if keep_date is False :
            df.drop(columns= ["date"], inplace= True)  
        
        if self.data == "traffic" :
            data_array = df.values[ : ,  : -1]
            OT_array = df.values[ : , -1]
            data_array = np.insert(data_array, 0, OT_array, axis= 1)

        elif self.data == "ETT" :
            data_array = df.values[ : ,  : -1]
            OT_array = df.values[ : , -1]
            data_array = np.insert(data_array, 0, OT_array, axis= 1)
        
        elif self.data == "weather" :
            minute_cols = ["p", "T", "Tpot", "Tdew"]
            hourly_cols = ["rh", "VPmax", "VPact", "VPdef", "sh", "H2OC", "rho"]
            daily_cols = ["wv", "max. wv", "wd"]
            weekly_cols = ["rain", "raining", "SWDR", "PAR", "max. PAR", "Tlog"]
            minute_cols = ["OT"] + minute_cols
            re_cols = minute_cols + hourly_cols + daily_cols + weekly_cols
            df = df.reindex(columns= re_cols)
            data_array = df.values

        return data_array
    
    
    def res_align(self, test_pred, test_y):
        pred_df = pd.DataFrame(test_pred)
        test_df = pd.DataFrame(test_y)
        resample_column = {'ETT': [3, 4, 5, 6], 'traffic': list(range(401, 862)), 'weather': list(range(5, 21))}
        index = 0 
        for column in resample_column[self.data]:
            column_resample_range = list()
            feature_value = test_df[column].values
            flag = feature_value[index]
            for i, value in enumerate(feature_value):
                if value == flag:
                    continue
                else:
                    column_resample_range.append((index, i-1))
                    index = i
                    flag = value
            column_resample_range.append((index, len(feature_value)-1))
            index = 0

            for start, end in column_resample_range:
                pred_df[column][start:end+1] = np.mean(pred_df[column][start:end+1])
        test_pred = pred_df.values
        return test_pred
    
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
                wonan = inputArray[:, j][~np.isnan(inputArray[:, j])]
                normalised_array.append((wonan - np.mean(wonan)) / np.std(wonan))
                meanlist.append(np.mean(wonan))
                stdlist.append(np.std(wonan))
            return np.array(normalised_array).T, meanlist, stdlist
        else :
            normalised_array = []
            for j in range(inputArray.shape[1]):
                normalised_array.append((inputArray[:, j] - meanlist[j]) / stdlist[j])
            return np.array(normalised_array).T, meanlist, stdlist


    def encoding(self, data_array_encode):
        print("fitting input size : ", data_array_encode.shape)
        model = MF_CLR(
            input_dims= self.grain_split_list[0], 
            grain_split= self.grain_split_list, 
            total_dim= self.total_dim,
            output_dims= 64,
            hidden_dims= 128,
            depth= 4,
            ph_dim= 32,
            device= "cpu",
            batch_size= 32,
            mask= True,
        )
        if self.training is True :
            loss_log = model.fit(
                train_data= data_array_encode,
                n_epochs= self.epoch,
                verbose= True,
            )
            model.save("results/models/MFCLR/" + self.test_file.split("_")[0] + "_" + str(self.mask_ratio) + ".pkl")
        else :
            model.load("results/models/MFCLR/" + self.test_file.split("_")[0] + "_" + str(self.mask_ratio) + ".pkl")

        print("encoding input size : ", data_array_encode.shape)
        padding = 0
        all_repr = model.encode(
            data_array_encode, 
            casual= False, 
            sliding_length= None, 
            sliding_padding= padding,
        )
        print("origin train_repr.shape : ", all_repr.shape)
        all_repr = all_repr.reshape(-1, all_repr.shape[-1])
        print("all_repr.shape : ", all_repr.shape)
        print()
        return all_repr
    

    def dataset_construct(self, inputArray, all_repr, labelArray, OT_app= True, _shuffle= True):
        min_length = min(inputArray.shape[0], all_repr.shape[0], labelArray.shape[0])
        inputArray = inputArray[-min_length : ]
        all_repr = all_repr[-min_length : ]
        labelArray = labelArray[-min_length : ]
        print("truncated size : ", inputArray.shape, all_repr.shape, labelArray.shape)
        test_T = int(inputArray.shape[0] * self.test_ratio)
        train_T = inputArray.shape[0] - test_T

        train_num = train_T  // self.win_len
        train_x, train_y = [], []
        for i in range(train_num) :
            x_start = train_T - (train_num - i) * self.win_len
            x_end = x_start + self.win_len
            if OT_app is True :
                this_x = all_repr[x_start : x_end, : ]
                this_OT = inputArray[x_start : x_end, : ]
                this_x = np.concatenate((this_x, this_OT), axis= 1).reshape(-1, )
                this_x = this_x.reshape(-1, 1)
                train_x.append(this_x)
            else :
                this_x = all_repr[x_start : x_end, : ]
                train_x.append(this_x)
            train_y.append(labelArray[x_start : x_end, : ])
        
        test_num = int(np.ceil(test_T / self.win_len))
        test_x, test_y = [], []
        for i in range(test_num) :
            x_start = inputArray.shape[0] - (test_num - i) * self.win_len
            x_end = x_start + self.win_len
            if OT_app is True :
                this_x = all_repr[x_start : x_end, : ]
                this_OT = inputArray[x_start : x_end, : ]
                this_x = np.concatenate((this_x, this_OT), axis= 1)
                this_x = this_x.reshape(-1, 1)
                test_x.append(this_x)
            else :
                test_x.append(all_repr[x_start : x_end, : ])
            test_y.append(labelArray[x_start : x_end, : ])
        print("last : ", x_start, x_end)
        
        train_x = np.nan_to_num(np.array(train_x))
        train_y = np.nan_to_num(np.array(train_y)).squeeze()
        test_x = np.nan_to_num(np.array(test_x))
        test_y = np.nan_to_num(np.array(test_y)).squeeze()
        print("window number : ", train_num, test_num)
        print("original input size : ", inputArray.shape)
        print("test needed : ", test_T)
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
        if train_features.shape[0] > MAX_SAMPLES :
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
    

    def imputation(self):
        label_array = self.gen_enc_array(df= self.df)
        label_array_norm, mean, std = self.normalisation(label_array)
        data_array = self.masking(label_array, self.mask_idx, 0)
        data_array_norm= self.masking(label_array_norm, self.mask_idx, 0)
        
        print("train shape : ", data_array.shape, data_array_norm.shape, data_array_norm[0, -6 : ])
        print("label shape : ", label_array.shape, label_array_norm.shape, label_array_norm[0, -6 : ])


        data_array_enc = self.data_split(data_array_norm)
        all_repr = self.encoding(data_array_enc)

        train_x, train_y, test_x, test_y = self.dataset_construct(data_array_norm, all_repr, label_array_norm)
        print()
        print(test_y[-2 : , : , 0])


        x_train, x_valid = train_x[ : int(train_x.shape[0] * 0.9), : ], train_x[int(train_x.shape[0] * 0.9) : , : ]
        y_train, y_valid = train_y[ : int(train_y.shape[0] * 0.9), : ], train_y[int(train_y.shape[0] * 0.9) : , : ]
        

        t = time.time()

        # forward forecasting
        x_train_forward = x_train[ : , : int(np.ceil(x_train.shape[1] / 2)), : ]
        x_valid_forward = x_valid[ : , : int(np.ceil(x_valid.shape[1] / 2)), : ]
        y_train_forward = y_train[ : , : int(np.ceil(y_train.shape[1] / 2)), : ]
        y_valid_forward = y_valid[ : , : int(np.ceil(y_valid.shape[1] / 2)), : ]
        x_train_forward = x_train_forward.reshape(x_train_forward.shape[0], -1)
        y_train_forward = y_train_forward.reshape(y_train_forward.shape[0], -1)
        x_valid_forward = x_valid_forward.reshape(x_valid_forward.shape[0], -1)
        y_valid_forward = y_valid_forward.reshape(y_valid_forward.shape[0], -1)
        print("x_train_forward.shape : ", x_train_forward.shape)
        print("y_train_forward.shape : ", y_train_forward.shape)
        print("x_valid_forward.shape : ", x_valid_forward.shape)
        print("y_valid_forward.shape : ", y_valid_forward.shape)
        predict_model_forward = self.fit_ridge(x_train_forward, y_train_forward, x_valid_forward, y_valid_forward)

        # backward forecasting
        x_train_backward = x_train[ : , -int(np.ceil(x_train.shape[1] / 2)) : , : ]
        x_valid_backward = x_valid[ : , -int(np.ceil(x_valid.shape[1] / 2)) : , : ]
        y_train_backward = y_train[ : , -int(np.ceil(y_train.shape[1] / 2)) : , : ]
        y_valid_backward = y_valid[ : , -int(np.ceil(y_valid.shape[1] / 2)) : , : ]
        x_train_backward = x_train_backward.reshape(x_train_backward.shape[0], -1)
        y_train_backward = y_train_backward.reshape(y_train_backward.shape[0], -1)
        x_valid_backward = x_valid_backward.reshape(x_valid_backward.shape[0], -1)
        y_valid_backward = y_valid_backward.reshape(y_valid_backward.shape[0], -1)
        print("x_train_backward.shape : ", x_train_backward.shape)
        print("y_train_backward.shape : ", y_train_backward.shape)
        print("x_valid_backward.shape : ", x_valid_backward.shape)
        print("y_valid_backward.shape : ", y_valid_backward.shape)
        predict_model_backward = self.fit_ridge(x_train_backward, y_train_backward, x_valid_backward, y_valid_backward)


        # forward forecasting
        test_x_forward = test_x[ : , : int(np.ceil(test_x.shape[1] / 2)), : ]
        test_x_forward = test_x_forward.reshape(test_x_forward.shape[0], -1)
        test_pred_forward = predict_model_forward.predict(test_x_forward)

        # backward forecasting
        test_x_backward = test_x[ : , -int(np.ceil(test_x.shape[1] / 2)) : , : ]
        test_x_backward = test_x_backward.reshape(test_x_backward.shape[0], -1)
        test_pred_backward = predict_model_backward.predict(test_x_backward)

        test_pred_forward = test_pred_forward.reshape(test_y.shape[0], int(np.ceil(self.win_len / 2)), test_y.shape[-1])
        test_pred_backward = test_pred_backward.reshape(test_y.shape[0], int(np.ceil(self.win_len / 2)), test_y.shape[-1])
        test_pred_left = test_pred_forward[ : , : int(np.floor(self.win_len / 2)), : ]
        test_pred_right = test_pred_backward[ : , -int(np.floor(self.win_len / 2)) : , : ]
        test_pred_centre = (test_pred_forward[ : , -1, : ] + test_pred_backward[ : , 0, : ]) / 2
        test_pred_centre = test_pred_centre.reshape(test_pred_centre.shape[0], 1, test_pred_centre.shape[1])
        print(test_pred_left.shape, test_pred_centre.shape, test_pred_right.shape)
        test_pred = np.concatenate([test_pred_left, test_pred_centre, test_pred_right], axis= 1)
        print(test_pred_forward.shape, test_pred_backward.shape, test_pred.shape)
        fit_time = time.time() - t
        print("fit time : ", fit_time)
        print(test_pred.shape, test_y.shape)
        
        test_pred = test_pred.reshape(test_y.shape)
        test_pred = test_pred.reshape(-1, test_pred.shape[2])
        test_y = test_y.reshape(-1, test_y.shape[2])

        test_pred = self.res_align(test_pred, test_y)
        mask_idx = self.mask_idx[self.mask_idx > data_array.shape[0] - test_y.shape[0]] - (data_array.shape[0] - test_y.shape[0])
        mask_idx = mask_idx[mask_idx >= 0]
        mse, mae, mape  = self.cal_loss(test_pred[mask_idx].reshape(-1, ), test_y[mask_idx].reshape(-1, ))
        print("normalised : ", mse, mae, mape)
        print("mean : ", mean[self.y_dim], "std : ", std[self.y_dim])

        return mse, mae




class Public_Dataset_TSCLR():
    def __init__(self, args):
        assert args.dataset in ["traffic", "ETTm1", "ETTm2", "weather"]
        assert args.ds in ["traffic", "ETT", "weather"]
        self.origin_file = args.dataset
        if args.dataset in ["ETTm1", "ETTm2"] :
            self.test_file = args.dataset + "_processed_OT=quarterly.csv"
        elif args.dataset == "traffic" :
            self.test_file = args.dataset + "_processed_OT=hourly.csv"
        elif args.dataset == "weather" :
            self.test_file = args.dataset + "_processed_OT=10-minute.csv"
        self.clr_method = args.method
        self.data = args.ds
        self.mask_ratio = args.mask_rate
        self.enc_len = args.enc_len
        self.win_len = args.window
        self.epoch = args.epochs
        self.test_ratio = args.split
        self.training = args.is_training
        self.saving = args.save
        if self.saving is True:
            model_file = self.clr_method.replace("-","")
            os.makedirs(f"results/models/{model_file}/", exist_ok=True)
        if args.ds == "traffic" :
            self.y_dim = 0
            self.total_dim = 862
        elif args.ds == "ETT" :
            self.y_dim = 0
            self.total_dim = 7
        elif args.ds == "weather" :
            self.y_dim = 0
            self.total_dim = 21
        self.df, self.mask_df, self.mask_idx = self.read_file()

    def masking(self, input_array, mask_idx, mask_val= np.nan):
        assert mask_val in [0, np.nan]
        df_mask = input_array.copy()
        for r in mask_idx :
            df_mask[r, : ] = mask_val
        return df_mask
    

    def read_file(self, mask_val= np.nan):
        assert mask_val in [0, np.nan]
        if self.data == "traffic" :
            df_ori, mask_idx = traffic_processing("hourly", self.mask_ratio)
        elif self.data == "ETT" :
            df_ori, mask_idx = ETT_processing(self.origin_file, "quarterly", self.mask_ratio)
        elif self.data == "weather" :
            df_ori, mask_idx = weather_processing("10-minute", self.mask_ratio)
        df_mask = df_ori.copy()
        for r in mask_idx :
            df_mask.iloc[r, : ] = mask_val
        return df_ori, df_mask, mask_idx
    

    def gen_enc_array(self, keep_date= False, df= None):
        if df is None :
            df = self.df_mask.copy()
        if keep_date is False :
            df.drop(columns= ["date"], inplace= True)  
        
        if self.data == "traffic" :
            data_array = df.values[ : ,  : -1]
            OT_array = df.values[ : , -1]
            data_array = np.insert(data_array, 0, OT_array, axis= 1)

        elif self.data == "ETT" :
            data_array = df.values[ : ,  : -1]
            OT_array = df.values[ : , -1]
            data_array = np.insert(data_array, 0, OT_array, axis= 1)
        
        elif self.data == "weather" :
            minute_cols = ["p", "T", "Tpot", "Tdew"]
            hourly_cols = ["rh", "VPmax", "VPact", "VPdef", "sh", "H2OC", "rho"]
            daily_cols = ["wv", "max. wv", "wd"]
            weekly_cols = ["rain", "raining", "SWDR", "PAR", "max. PAR", "Tlog"]
            minute_cols = ["OT"] + minute_cols
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
                wonan = inputArray[:, j][~np.isnan(inputArray[:, j])]
                normalised_array.append((wonan - np.mean(wonan)) / np.std(wonan))
                meanlist.append(np.mean(wonan))
                stdlist.append(np.std(wonan))
            return np.array(normalised_array).T, meanlist, stdlist
        else :
            normalised_array = []
            for j in range(inputArray.shape[1]):
                normalised_array.append((inputArray[:, j] - meanlist[j]) / stdlist[j])
            return np.array(normalised_array).T, meanlist, stdlist
    


    def TLoss_encoding(self, data_array_encode, test_array):
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
            loss_log = model.fit(data_array_encode)
            if self.saving is True :
                model.save_model("results/models/TLoss/" + self.test_file.split("_")[0] + "_" + str(self.mask_ratio) + ".pkl")
        else :
            try :
                model.save_model("results/models/TLoss/" + self.test_file.split("_")[0] + "_" + str(self.mask_ratio) + ".pkl")
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
    

    def TSTCC_encoding(self, data_array_encode):      
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
            model.save_model("results/models/TSTCC/" + self.test_file.split("_")[0] + "_" + str(self.mask_ratio) + ".pkl")
        else :
            model.load_model("results/models/TSTCC/" + self.test_file.split("_")[0] + "_" + str(self.mask_ratio) + ".pkl")
            
        train_repr = model.encode(data_array_encode)
        print("original train_repr.shape : ", train_repr.shape)
        all_repr = train_repr.reshape(-1, train_repr.shape[-1])
        print("all_repr.shape : ", all_repr.shape)
        print()
        return all_repr
    

    def TNC_encoding(self, data_array_encode):   
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
            model.save("results/models/TNC/" + self.test_file.split("_")[0] + "_" + str(self.mask_ratio) + ".pkl")
        else :
            model.load("results/models/TNC/" + self.test_file.split("_")[0] + "_" + str(self.mask_ratio) + ".pkl")
        
        train_repr = model.encode(data_array_encode)
        print("original train_repr.shape : ", train_repr.shape)
        all_repr = train_repr.reshape(-1, train_repr.shape[-1])
        print("all_repr.shape : ", all_repr.shape)
        print()
        return all_repr


    def TS2Vec_encoding(self, data_array_encode):
        print("encoding input size : ", data_array_encode.shape)
        model = TS2Vec(
            input_dims= data_array_encode.shape[-1], 
            device= "cpu",
        )
        if self.training is True :
            loss_log = model.fit(train_data= data_array_encode,
                            n_epochs= self.epoch,
                            verbose= True)
            model.save("results/models/TS2Vec/" + self.test_file.split("_")[0] + "_" + str(self.mask_ratio) + ".pkl")
        else :
            model.load("results/models/TS2Vec/" + self.test_file.split("_")[0] + "_" + str(self.mask_ratio) + ".pkl")

        train_repr = model.encode(data_array_encode)
        print("original train_repr.shape : ", train_repr.shape)
        all_repr = train_repr.reshape(-1, train_repr.shape[-1])
        print("all_repr.shape : ", all_repr.shape)
        print()
        return all_repr
    

    def CPC_encoding(self, data_array_encode):
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
            model.save_model(fn= "results/models/CPC/" + self.test_file.split("_")[0] + "_" + str(self.mask_ratio) + ".pkl")
        else :
            model.load_model("results/models/CPC/" + self.test_file.split("_")[0] + "_" + str(self.mask_ratio) + ".pkl")

        train_repr = model.encode(data_array_encode)
        print("original train_repr.shape : ", train_repr.shape)
        all_repr = train_repr.reshape(-1, train_repr.shape[-1])
        print("all_repr.shape : ", all_repr.shape)
        print()
        return all_repr
    

    def CoST_encoding(self, data_array_encode):
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
            model.save("results/models/CoST/" + self.test_file.split("_")[0] + "_" + str(self.mask_ratio) + ".pkl")
        else :
            model.load("results/models/CoST/" + self.test_file.split("_")[0] + "_" + str(self.mask_ratio) + ".pkl")

        train_repr = model.encode(data_array_encode)
        print("original train_repr.shape : ", train_repr.shape)
        all_repr = train_repr.reshape(-1, train_repr.shape[-1])
        print("all_repr.shape : ", all_repr.shape)
        print()
        return all_repr
    

    def TFC_encoding(self, data_array_encode):
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
            model.save("results/models/TFC/" + self.test_file.split("_")[0] + "_" + str(self.mask_ratio) + ".pkl")
        else :
            model.load("results/models/TFC/" + self.test_file.split("_")[0] + "_" + str(self.mask_ratio) + ".pkl")

        train_repr = model.encode(data_array_encode)
        print("original train_repr.shape : ", train_repr.shape)
        all_repr = train_repr.reshape(-1, train_repr.shape[-1])
        print("all_repr.shape : ", all_repr.shape)
        print()
        return all_repr
    


    def dataset_construct(self, inputArray, all_repr, labelArray, OT_app= True, _shuffle= True):
        min_length = min(inputArray.shape[0], all_repr.shape[0], labelArray.shape[0])
        inputArray = inputArray[-min_length : ]
        all_repr = all_repr[-min_length : ]
        labelArray = labelArray[-min_length : ]
        print("truncated size : ", inputArray.shape, all_repr.shape, labelArray.shape)
        test_T = int(inputArray.shape[0] * self.test_ratio)
        train_T = inputArray.shape[0] - test_T

        train_num = train_T  // self.win_len
        train_x, train_y = [], []
        for i in range(train_num) :
            x_start = train_T - (train_num - i) * self.win_len
            x_end = x_start + self.win_len
            if OT_app is True :
                this_x = all_repr[x_start : x_end, : ]
                this_OT = inputArray[x_start : x_end, : ]
                this_x = np.concatenate((this_x, this_OT), axis= 1).reshape(-1, 1)
                train_x.append(this_x)
            else :
                train_x.append(all_repr[x_start : x_end, : ])
            train_y.append(labelArray[x_start : x_end, : ])
        
        test_num = int(np.ceil(test_T / self.win_len))
        test_x, test_y = [], []
        for i in range(test_num) :
            x_start = inputArray.shape[0] - (test_num - i) * self.win_len
            x_end = x_start + self.win_len
            if OT_app is True :
                this_x = all_repr[x_start : x_end, : ]
                this_OT = inputArray[x_start : x_end, : ]
                this_x = np.concatenate((this_x, this_OT), axis= 1).reshape(-1, 1)
                test_x.append(this_x)
            else :
                test_x.append(all_repr[x_start : x_end, : ])
            test_y.append(labelArray[x_start : x_end, : ])
        
        train_x = np.nan_to_num(np.array(train_x))
        train_y = np.nan_to_num(np.array(train_y)).squeeze()
        test_x = np.nan_to_num(np.array(test_x))
        test_y = np.nan_to_num(np.array(test_y)).squeeze()
        print("window number : ", train_num, test_num)
        print("original input size : ", inputArray.shape)
        print("test needed : ", test_T)
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
        if train_features.shape[0] > MAX_SAMPLES :
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
    

    def imputation(self):
        label_array = self.gen_enc_array(df= self.df)
        label_array_norm, mean, std = self.normalisation(label_array)
        data_array = self.masking(label_array, self.mask_idx, 0)
        data_array_norm= self.masking(label_array_norm, self.mask_idx, 0)

        print("train shape : ", data_array.shape, data_array_norm.shape, data_array_norm[0, -6 : ])
        print("label shape : ", label_array.shape, label_array_norm.shape, label_array_norm[0, -6 : ])


        if self.clr_method == "T-Loss" :
            data_array_enc = self.data_split(data_array_norm)
            all_repr = self.TLoss_encoding(data_array_enc)
        elif self.clr_method == "TS-TCC" :
            data_array_enc = self.data_split(data_array_norm, "cut")
            all_repr = self.TSTCC_encoding(data_array_enc)
        elif self.clr_method == "TNC" :
            data_array_enc = self.data_split(data_array_norm)
            all_repr = self.TNC_encoding(data_array_enc)
        elif self.clr_method == "TS2Vec" :
            data_array_enc = self.data_split(data_array_norm, "cut")
            all_repr = self.TS2Vec_encoding(data_array_enc)
        elif self.clr_method == "CPC" :
            data_array_enc = self.data_split(data_array_norm, "cut")
            all_repr = self.CPC_encoding(data_array_enc)
        elif self.clr_method == "CoST" :
            data_array_enc = self.data_split(data_array_norm, "cut")
            all_repr = self.CoST_encoding(data_array_enc)
        elif self.clr_method == "TF-C" :
            data_array_enc = self.data_split(data_array_norm)
            all_repr = self.TFC_encoding(data_array_enc)
        else : raise


        train_x, train_y, test_x, test_y = self.dataset_construct(data_array_norm, all_repr, label_array_norm)
        print()

        x_train, x_valid = train_x[ : int(train_x.shape[0] * 0.9), : ], train_x[int(train_x.shape[0] * 0.9) : , : ]
        y_train, y_valid = train_y[ : int(train_y.shape[0] * 0.9), : ], train_y[int(train_y.shape[0] * 0.9) : , : ]
        
        # forward forecasting
        x_train_forward = x_train[ : , : int(np.ceil(x_train.shape[1] / 2)), : ]
        x_valid_forward = x_valid[ : , : int(np.ceil(x_valid.shape[1] / 2)), : ]
        y_train_forward = y_train[ : , : int(np.ceil(y_train.shape[1] / 2)), : ]
        y_valid_forward = y_valid[ : , : int(np.ceil(y_valid.shape[1] / 2)), : ]
        x_train_forward = x_train_forward.reshape(x_train_forward.shape[0], -1)
        y_train_forward = y_train_forward.reshape(y_train_forward.shape[0], -1)
        x_valid_forward = x_valid_forward.reshape(x_valid_forward.shape[0], -1)
        y_valid_forward = y_valid_forward.reshape(y_valid_forward.shape[0], -1)
        print("x_train_forward.shape : ", x_train_forward.shape)
        print("y_train_forward.shape : ", y_train_forward.shape)
        print("x_valid_forward.shape : ", x_valid_forward.shape)
        print("y_valid_forward.shape : ", y_valid_forward.shape)
        predict_model_forward = self.fit_ridge(x_train_forward, y_train_forward, x_valid_forward, y_valid_forward)

        # backward forecasting
        x_train_backward = x_train[ : , -int(np.ceil(x_train.shape[1] / 2)) : , : ]
        x_valid_backward = x_valid[ : , -int(np.ceil(x_valid.shape[1] / 2)) : , : ]
        y_train_backward = y_train[ : , -int(np.ceil(y_train.shape[1] / 2)) : , : ]
        y_valid_backward = y_valid[ : , -int(np.ceil(y_valid.shape[1] / 2)) : , : ]
        x_train_backward = x_train_backward.reshape(x_train_backward.shape[0], -1)
        y_train_backward = y_train_backward.reshape(y_train_backward.shape[0], -1)
        x_valid_backward = x_valid_backward.reshape(x_valid_backward.shape[0], -1)
        y_valid_backward = y_valid_backward.reshape(y_valid_backward.shape[0], -1)
        print("x_train_backward.shape : ", x_train_backward.shape)
        print("y_train_backward.shape : ", y_train_backward.shape)
        print("x_valid_backward.shape : ", x_valid_backward.shape)
        print("y_valid_backward.shape : ", y_valid_backward.shape)
        predict_model_backward = self.fit_ridge(x_train_backward, y_train_backward, x_valid_backward, y_valid_backward)

        t = time.time()

        # forward forecasting
        test_x_forward = test_x[ : , : int(np.ceil(test_x.shape[1] / 2)), : ]
        test_x_forward = test_x_forward.reshape(test_x_forward.shape[0], -1)
        test_pred_forward = predict_model_forward.predict(test_x_forward)

        # backward forecasting
        test_x_backward = test_x[ : , -int(np.ceil(test_x.shape[1] / 2)) : , : ]
        test_x_backward = test_x_backward.reshape(test_x_backward.shape[0], -1)
        test_pred_backward = predict_model_backward.predict(test_x_backward)

        test_pred_forward = test_pred_forward.reshape(test_y.shape[0], int(np.ceil(self.win_len / 2)), test_y.shape[-1])
        test_pred_backward = test_pred_backward.reshape(test_y.shape[0], int(np.ceil(self.win_len / 2)), test_y.shape[-1])
        test_pred_left = test_pred_forward[ : , : int(np.floor(self.win_len / 2)), : ]
        test_pred_right = test_pred_backward[ : , -int(np.floor(self.win_len / 2)) : , : ]
        test_pred_centre = (test_pred_forward[ : , -1, : ] + test_pred_backward[ : , 0, : ]) / 2
        test_pred_centre = test_pred_centre.reshape(test_pred_centre.shape[0], 1, test_pred_centre.shape[1])
        print(test_pred_left.shape, test_pred_centre.shape, test_pred_right.shape)
        test_pred = np.concatenate([test_pred_left, test_pred_centre, test_pred_right], axis= 1)
        print(test_pred_forward.shape, test_pred_backward.shape, test_pred.shape)
        fit_time = time.time() - t
        print("fit time : ", fit_time)
        print(test_pred.shape, test_y.shape)
        
        test_pred = test_pred.reshape(test_y.shape)
        test_pred = test_pred.reshape(-1, test_pred.shape[2])
        test_y = test_y.reshape(-1, test_y.shape[2])

        print(test_pred.shape, test_y.shape)

        mask_idx = self.mask_idx[self.mask_idx > data_array.shape[0] - test_y.shape[0]] - (data_array.shape[0] - test_y.shape[0])
        mask_idx = mask_idx[mask_idx >= 0]
        mse, mae, mape  = self.cal_loss(test_pred[mask_idx].reshape(-1, ), test_y[mask_idx].reshape(-1, ))
        print("normalised : ", mse, mae, mape)
        print("mean : ", mean[self.y_dim], "std : ", std[self.y_dim])
        return mse, mae






if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_training', type=bool, required=True, default=1, help='status')
    parser.add_argument('--dataset', type=str, required=True, help='The dataset name, This can be set to ETTm1, ETTm2, traffic, weather')
    parser.add_argument('--method', type=str, required=True, default='mf_clr', help='model_name, options:[MF_CLR, T-Loss, TS-TCC, TNC, TS2Vec, CPC, CoST]')
    parser.add_argument('--use_gpu', type=bool, default=False, help='use_gpu')
    parser.add_argument('--batch_size', type=int, default=32, help='The batch size(defaults to 16)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate(defaluts to 0.001)')
    parser.add_argument('--epochs', type=int, default=30, help='The number of epochs(defaults to 30)')
    parser.add_argument('--enc_len', type=int, default=128, help='input sequence length')
    parser.add_argument('--ot_granu', type=str, default='quarterly', help='frequency of the forecast target')
    parser.add_argument('--mask_rate', type=float, default='0.25', help='mask ratio')
    parser.add_argument('--save', type=bool, default=True, help='save the checkpoint')
    parser.add_argument('--window', type=int, default=21, help='window')
    parser.add_argument('--split', type=float, default=0.2, help='split')
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
        torch.cuda.set_device(args.gpu)
    
    print('Args in experiment:')
    print(args)

    if args.dataset in ["ETTm1", "ETTm2"] : 
        args.ds = "ETT"
    else :
        args.ds = args.dataset

    CLR_EXP = Public_Dataset_MFCLR(args)  if args.method=='MF-CLR' else Public_Dataset_TSCLR(args)
 
    CLR_EXP.imputation()
    print(args.method, ['whole'], args.dataset, args.mask_rate)
    print()
    print()
    print()
    print()