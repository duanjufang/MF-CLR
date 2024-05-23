import os
import time
import torch
import argparse
import warnings
import numpy as np
import pandas as pd
from data.dataset_preprocessing_AD import resample_AD
from MFCLR import MF_CLR 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from algos.TLoss.interface import Unsupervided_Scalable as TLoss
from algos.TS_TCC.interface import TS_TCC
from algos.TNC.interface import TNC
from algos.TS2Vec.ts2vec import TS2Vec
from algos.contrastive_predictive_coding.interface import Contrasive as CPC
from algos.CosT.cost import CoST
from algos.TFC.code.TFC.interface import TFC_inter
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, precision_recall_fscore_support


warnings.filterwarnings('ignore')
abs_path = os.path.dirname(os.path.realpath(__file__))




class Public_Dataset_AD():
    
    def __init__(self,args):
        assert args.dataset in ["SWaT", "SMD", "SMAP", "MSL"]
        self.data = args.dataset
        self.method = args.method
        self.enc_len = args.enc_len
        self.win_len = args.window
        self.epoch = args.epochs
        self.training = args.is_training
        self.saving = args.save
        self.learning_rate = args.lr
        self.projector_dim = args.ph_dim
        self.batch_size = args.batch_size
        if args.use_gpu is False:
            self.device = 'cpu'
        else:
            self.device = 'cuda'
        if self.saving is True:
            model_file = self.method.replace("-","")
            os.makedirs(f"results/models/{model_file}/", exist_ok=True)
        self.output_dims = args.out_dim
        self.hidden_dims = args.hidden_dim
        self.depth = args.depth
        if args.dataset == "MSL" :
            self.grain_split_list = [1, 28, 55]
        elif args.dataset == "SMAP" :
            self.grain_split_list = [1, 13, 25]
        elif args.dataset == "SMD" :
            self.grain_split_list = [13, 26, 38]
        elif args.dataset == "SWaT" :
            self.grain_split_list = [25, 34, 51]
        self.total_dim = self.grain_split_list[-1]
    

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


    def encoding(self, data_array_encode, test_array, sub_idx:int):
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
                model.save("results/models/MFCLR/" + self.data + "_" + str(sub_idx) + ".pkl")
        else :
            try :
                model.load("results/models/MFCLR/" + self.data + "_" + str(sub_idx) + ".pkl")
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
        return train_repr, test_repr
    

    def TLoss_encoding(self, data_array_encode, test_array, sub_idx:int):
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
        if self.training is True :
            loss_log = model.fit(data_array_encode)
            if self.saving is True :
                model.save("results/models/TLoss/" + self.data + "_" + str(sub_idx) + ".pkl")
        else :
            try :
                model.load("results/models/TLoss/" + self.data + "_" + str(sub_idx) + ".pkl")
            except :
                loss_log = model.fit(data_array_encode)
        
        train_repr = model.encode(data_array_encode)
        print("original train_repr.shape : ", train_repr.shape)
        train_repr = train_repr.reshape(-1, train_repr.shape[-1])
        print("train_repr.shape : ", train_repr.shape)
        test_repr = model.encode(self.data_split(test_array, "slide"))
        test_repr = test_repr.reshape(-1, test_repr.shape[-1])
        print("test_repr.shape : ", test_repr.shape)
        return train_repr, test_repr
    

    def TSTCC_encoding(self, data_array_encode, test_array, sub_idx:int):     
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
                model.save_model("results/models/TSTCC/" + self.data + "_" + str(sub_idx) + ".pkl")
        else :
            try :
                model.load_model("results/models/TSTCC/" + self.data + "_" + str(sub_idx) + ".pkl")
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
        return train_repr, test_repr
    

    def TNC_encoding(self, data_array_encode, test_array, sub_idx:int):
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
                model.save("results/models/TNC/" + self.data + "_" + str(sub_idx) + ".pkl")
        else :
            try :
                model.load("results/models/TNC/" + self.data + "_" + str(sub_idx) + ".pkl")
            except :
                loss, accuracy = model.fit(data_array_encode, self.epoch)
        
        train_repr = model.encode(data_array_encode)
        print("original train_repr.shape : ", train_repr.shape)
        train_repr = train_repr.reshape(-1, train_repr.shape[-1])
        print("train_repr.shape : ", train_repr.shape)
        test_repr = model.encode(self.data_split(test_array, "slide"))
        test_repr = test_repr.reshape(-1, test_repr.shape[-1])
        print("test_repr.shape : ", test_repr.shape)
        return train_repr, test_repr


    def TS2Vec_encoding(self, data_array_encode, test_array, sub_idx:int):
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
                model.save("results/models/TS2Vec/" + self.data + "_" + str(sub_idx) + ".pkl")
        else :
            try :
                model.load("results/models/TS2Vec/" + self.data + "_" + str(sub_idx) + ".pkl")
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
        return train_repr, test_repr
    

    def CPC_encoding(self, data_array_encode, test_array, sub_idx:int):
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
                model.save_model(fn= "results/models/CPC/" + self.data + "_" + str(sub_idx) + ".pkl")
        else :
            try :
                model.load_model("results/models/CPC/" + self.data + "_" + str(sub_idx) + ".pkl")
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
        return train_repr, test_repr
    

    def CoST_encoding(self, data_array_encode, test_array, sub_idx:int):
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
                model.save("results/models/CoST/" + self.data + "_" + str(sub_idx) + ".pkl")
        else :
            try :
                model.load("results/models/CoST/" + self.data + "_" + str(sub_idx) + ".pkl")
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
        return train_repr, test_repr
    

    def TFC_encoding(self, data_array_encode, test_array, sub_idx:int):
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
                model.save("results/models/TFC/" + self.data + "_" + str(sub_idx) + ".pkl")
        else :
            try :
                model.load("results/models/TFC/" + self.data + "_" + str(sub_idx) + ".pkl")
            except :
                model.fit(data_array_encode, self.epoch)

        train_repr = model.encode(data_array_encode)
        print("original train_repr.shape : ", train_repr.shape)
        train_repr = train_repr.reshape(-1, train_repr.shape[-1])
        print("train_repr.shape : ", train_repr.shape)
        test_array_encode = self.data_split(test_array, "slide")
        print("test_array_encode.shape : ", test_array_encode.shape)
        test_repr = model.encode(test_array_encode)
        test_repr = test_repr.reshape(-1, test_repr.shape[-1])
        print("test_repr.shape : ", test_repr.shape)
        return train_repr, test_repr
    

    def dataset_construct(self, inputArray, repr, val_app= False, _shuffle= True): 
        min_length = min(inputArray.shape[0], repr.shape[0])
        inputArray = inputArray[-min_length : ]
        repr = repr[-min_length : ]
        print("truncated size : ", inputArray.shape, repr.shape)


        win_num = int(inputArray.shape[0] // self.win_len)
        x, y = [], []
        for i in range(win_num):
            x_start = inputArray.shape[0] - (win_num - i) * self.win_len
            x_end = x_start + self.win_len
            this_x = repr[x_start : x_end, : ]
            if val_app is True :
                this_val = inputArray[x_start : x_end, : ]
                this_x = np.concatenate((this_x, this_val), axis= 1)
            x.append(this_x)
            y.append(inputArray[x_start : x_end, : ])
        
        x, y = np.array(x), np.array(y)
        print("window number : ", win_num)
        print("x shape : ", x.shape)
        print("y shape : ", y.shape)

        if _shuffle is True :
            random_idx = np.random.permutation(x.shape[0])
            x = x[random_idx]
            y = y[random_idx]
        return x, y


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
    

    def adjustment(self, gt, pred):
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1
        return gt, pred
    

    def cal_loss_clsf(self, pred_y, label_y):
        acc = accuracy_score(label_y, pred_y)
        precision, recall, F1, _ = precision_recall_fscore_support(label_y, pred_y, average='binary')
        return acc, precision, recall, F1
    

    def anomaly_detection(self, anomaly_ratio, exp_start= 0):
        train_X, test_X, test_label = resample_AD(self.data)
        assert len(train_X) == len(test_X) == len(test_label)
        total_acc, total_precision, total_recall, total_F1 = 0, 0, 0, 0
        for exp_idx in range(exp_start, len(train_X)):
            train_array = train_X[exp_idx]
            test_array = test_X[exp_idx]
            this_label = test_label[exp_idx]
            train_array_norm = train_array
            test_array_norm  = test_array
            print("train shape : ", train_array.shape, train_array_norm.shape)
            print(" test shape : ", test_array.shape, test_array_norm.shape)
            print("label shape : ", this_label.shape)


            if self.method == "MF-CLR" :
                data_array_enc = self.data_split(train_array_norm)
                train_repr, test_repr = self.encoding(data_array_enc, test_array_norm, exp_idx)
            elif self.method == "T-Loss" :
                data_array_enc = self.data_split(train_array_norm, "slide")
                train_repr, test_repr = self.TLoss_encoding(data_array_enc, test_array_norm, exp_idx)
            elif self.method == "TS-TCC" :
                data_array_enc = self.data_split(train_array_norm, "cut")
                train_repr, test_repr = self.TSTCC_encoding(data_array_enc, test_array_norm, exp_idx)
            elif self.method == "TNC" :
                data_array_enc = self.data_split(train_array_norm, "slide")
                train_repr, test_repr = self.TNC_encoding(data_array_enc, test_array_norm, exp_idx)
            elif self.method == "TS2Vec" :
                data_array_enc = self.data_split(train_array_norm, "cut")
                train_repr, test_repr = self.TS2Vec_encoding(data_array_enc, test_array_norm, exp_idx)
            elif self.method == "CPC" :
                data_array_enc = self.data_split(train_array_norm, "cut")
                train_repr, test_repr = self.CPC_encoding(data_array_enc, test_array_norm, exp_idx)
            elif self.method == "CoST" :
                data_array_enc = self.data_split(train_array_norm, "cut")
                train_repr, test_repr = self.CoST_encoding(data_array_enc, test_array_norm, exp_idx)
            elif self.method == "TF-C" :
                data_array_enc = self.data_split(train_array_norm, "slide")
                train_repr, test_repr = self.TFC_encoding(data_array_enc, test_array_norm, exp_idx)
            else : raise


            train_x, train_y = self.dataset_construct(train_array_norm, train_repr)
            test_x, test_y = self.dataset_construct(test_array_norm, test_repr)
            print()

            x_train, x_valid = train_x[ : int(train_x.shape[0] * 0.9), : ], train_x[int(train_x.shape[0] * 0.9) : , : ]
            y_train, y_valid = train_y[ : int(train_y.shape[0] * 0.9), : ], train_y[int(train_y.shape[0] * 0.9) : , : ]


            t = time.time()
            
            # self-forecasting training
            x_train = x_train.reshape(x_train.shape[0], -1)
            y_train = y_train.reshape(y_train.shape[0], -1)
            x_valid = x_valid.reshape(x_valid.shape[0], -1)
            y_valid = y_valid.reshape(y_valid.shape[0], -1)
            print("x_train.shape : ", x_train.shape)
            print("y_train.shape : ", y_train.shape)
            print("x_valid.shape : ", x_valid.shape)
            print("y_valid.shape : ", y_valid.shape)
            predict_model = self.fit_ridge(x_train, y_train, x_valid, y_valid)        


            # self forecasting
            y_train_hat = predict_model.predict(train_x.reshape(train_x.shape[0], -1))
            y_test_hat = predict_model.predict(test_x.reshape(test_x.shape[0], -1))
            y_train_hat = y_train_hat.reshape(train_y.shape)
            y_test_hat = y_test_hat.reshape(test_y.shape)
            train_y = train_y.reshape(-1, train_y.shape[-1])
            y_train_hat = y_train_hat.reshape(-1, y_train_hat.shape[-1])
            test_y = test_y.reshape(-1, test_y.shape[-1])
            y_test_hat = y_test_hat.reshape(-1, y_test_hat.shape[-1])
            print(train_y.shape, y_train_hat.shape, test_y.shape, y_test_hat.shape)
            train_mse_list = [mean_squared_error(train_y[i, :], y_train_hat[i, :]) for i in range(train_y.shape[0])]
            test_mse_list = [mean_squared_error(test_y[i, :], y_test_hat[i, :]) for i in range(test_y.shape[0])]
            print("MSE list : ", len(train_mse_list), len(test_mse_list))
            print()
            
            # find threshold
            total_energy = np.array(train_mse_list + test_mse_list)
            threshold = np.percentile(total_energy, 100 - anomaly_ratio)
            print("Threshold :", threshold)

            # prediction
            preds = (np.array(test_mse_list) > threshold).astype(int)
            gt = this_label[-preds.shape[0] : ].reshape(-1, )          
            print("pred:   ", preds.shape)
            print("gt:     ", gt.shape)
            gt, preds = self.adjustment(gt, preds)
            acc, precision, recall, F1 = self.cal_loss_clsf(preds, gt)
            print("anomaly detection results : ", self.data, exp_idx)
            print('Acc: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f' % (acc, precision, recall, F1))
            total_acc += acc
            total_precision += precision
            total_recall += recall
            total_F1 += F1
            print("accumulated results : ", exp_idx + 1, " rounds")
            print('Acc: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f' % (total_acc, total_precision, total_recall, total_F1))
            test_num = exp_idx + 1 - exp_start
            print("average : ")
            print('Acc: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f' % (
                total_acc / test_num, total_precision / test_num, total_recall / test_num, total_F1 / test_num
            ))
            print()
            print()
            print()
            print()
        
        temp_res_df = pd.DataFrame()
        pred = preds.reshape(-1, )
        label = gt.reshape(-1, )
        temp_res_df["pred"] = pred
        temp_res_df["label"] = label
        temp_res_df.to_csv("temp_res_df.csv", index= False)


if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description='MF-CLR')
    parser.add_argument('--is_training', type=bool, required=True, default=True, help='status')
    parser.add_argument('--dataset', type=str, required=True, help='The dataset name, This can be set to ETTm1, ETTm2, traffic, weather')
    parser.add_argument('--method', type=str, required=True, default='mf_clr', help='model_name, options:[MF_CLR, T-Loss, TS-TCC, TNC, TS2Vec, CPC, CoST]')
    parser.add_argument('--use_gpu', type=bool, default=False, help='use_gpu')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--batch_size', type=int, default=32, help='The batch size(defaults to 32)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate(defaluts to 0.001)')
    parser.add_argument('--epochs', type=int, default=30, help='The number of epochs(defaults to 30)')
    parser.add_argument('--enc_len', type=int, default=128, help='input sequence length')
    parser.add_argument('--ot_granu', type=str, default='quarterly', help='frequency of the forecast target')
    parser.add_argument('--mask_rate', type=str, default='0.25', help='mask ratio')
    parser.add_argument('--save', type=bool, default=True, help='save the checkpoint')
    parser.add_argument('--window', type=int, default=3, help='window size')
    parser.add_argument('--split', type=float, default=0.2, help='split')
    parser.add_argument('--anomaly_ratio', type=float, default=1, help='anomaly_ratio')
    parser.add_argument('--ph_dim', type=int, default=32, help='dimension of projector')
    parser.add_argument('--out_dim', type=int, default=64, help='dimension of output')
    parser.add_argument('--hidden_dim', type=int, default=128, help='dimension of hidden')
    parser.add_argument('--depth', type=int, default=4, help='depth')



    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)
    # ar = {
    #     "SWaT" : 0.5,
    #     "SMD" : 4,
    #     "MSL" : 4,
    #     "SMAP" : 1.5,
    # }

    CLR_EXP = Public_Dataset_AD(args)

    CLR_EXP.anomaly_detection(anomaly_ratio=args.anomaly_ratio)
