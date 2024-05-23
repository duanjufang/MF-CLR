import os
import torch
import argparse
import warnings
import numpy as np
import pandas as pd
from data.dataset_preprocessing_classification import UEA_resample
from MFCLR import MF_CLR
from algos.informer.interface import train_and_predict
from algos.TLoss.interface import Unsupervided_Scalable as TLoss
from algos.TS_TCC.interface import TS_TCC
from algos.TNC.interface import TNC
from algos.TS2Vec.ts2vec import TS2Vec
from algos.contrastive_predictive_coding.interface import Contrasive as CPC
from algos.CosT.cost import CoST
from algos.TFC.code.TFC.interface import TFC_inter
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, f1_score, recall_score


warnings.filterwarnings('ignore')
abs_path = os.path.dirname(os.path.realpath(__file__))




encoding_dict = {
    "ArticularyWordRecognition" : (24, [3, 6, 9]),
    "AtrialFibrillation" : (40, [1, 2]),
    "BasicMotions" : (25, [3, 6]),
    "CharacterTrajectories" : (30, [2, 3]),
    "Cricket" : (133, [3, 6]),
    "DuckDuckGeese" : (30, [500, 1000, 1345]),
    "EigenWorms" : (562, [3, 6]),
    "Epilepsy" : (34, [2, 3]),
    "EthanolConcentration" : (103, [2, 3]), 
    "ERing" : (13, [2, 4]), 
    "FaceDetection" : (20, [72, 144]), 
    "FingerMovements" : (16, [19, 28]), 
    "HandMovementDirection" : (40, [5, 8, 10]), 
    "Handwriting" : (30, [2, 3]), 
    "Heartbeat" : (45, [21, 41, 61]), 
    "InsectWingbeat" : (10, [100, 200]), 
    "JapaneseVowels" : (7, [6, 12]), 
    "Libras" : (15, [1, 2]), 
    "LSST" : (12, [2, 4, 6]), 
    "MotorImagery" : (100, [16, 32, 48, 64]), 
    "NATOPS" : (10, [12, 24]), 
    "PenDigits" : (4, [1, 2]), 
    "PEMS-SF" : (24, [321, 642, 963]), 
    "PhonemeSpectra" : (31, [6, 11]), 
    "RacketSports" : (30, [3, 6]), 
    "SelfRegulationSCP1" : (112, [2, 6]), 
    "SelfRegulationSCP2" : (144, [2, 7]), 
    "SpokenArabicDigits" : (31, [6, 13]), 
    "StandWalkJump" : (50, [2, 4]), 
    "UWaveGestureLibrary" : (21, [1, 2, 3]), 
}




class Public_Dataset_Classification():

    def __init__(self, args):
        assert args.encoding in ["whole", "cut", "slide"]
        self.clr_method = args.method
        self.epoch = args.epochs
        self.dataset = args.dataset
        self.enc_len = encoding_dict[args.dataset][0]
        self.grain_split_list = encoding_dict[args.dataset][1]
        self.total_dim = self.grain_split_list[-1]
        self.encoding = args.encoding
        self.training = args.is_training
        self.saving = args.save
        self.learning_rate = args.lr
        self.projector_dim = args.ph_dim
        self.batch_size = args.batch_size
        if self.saving is True:
            model_file = self.clr_method.replace("-","")
            os.makedirs(f"results/models/{model_file}/", exist_ok=True)
        if args.use_gpu is False:
            self.device = 'cpu'
        else:
            self.device = 'cuda'
        self.output_dims = args.out_dim
        self.hidden_dims = args.hidden_dim
        self.depth = args.depth
    

    def normalisation(self, inputArray, meanlist= None, stdlist= None):
        if meanlist is None and stdlist is None :
            meanlist, stdlist = [], []
            normalised_array = []
            for j in range(inputArray.shape[1]):
                normalised_array.append((inputArray[:, j] - np.mean(inputArray[:, j])) / np.std(inputArray[:, j]))
                meanlist.append(np.mean(inputArray[:, j]))
                stdlist.append(np.std(inputArray[:, j]))
            return np.array(normalised_array).transpose(1, 0, 2), meanlist, stdlist
        else :
            normalised_array = []
            for j in range(inputArray.shape[1]):
                normalised_array.append((inputArray[:, j] - meanlist[j]) / stdlist[j])
            return np.array(normalised_array).transpose(1, 0, 2), meanlist, stdlist
    

    def data_split(self, data_array_norm, _method= "slide"):
        # input [N, T, D]
        # return [sample, subseq, T, D]
        assert _method in ["slide", "cut"]
        if _method == "cut" :
            cut_rows = int(np.floor(data_array_norm.shape[1] / self.enc_len)) * self.enc_len
            data_array_norm = data_array_norm[ : , -cut_rows : , : ]
            win_num = int(data_array_norm.shape[1] // self.enc_len)
            split_data = np.apply_along_axis(lambda x : np.split(x, win_num), 1, data_array_norm)
            return split_data
        elif _method == "slide" :
            win_num = data_array_norm.shape[1] - self.enc_len + 1
            split_data = np.apply_along_axis(lambda x : [x[i : i + self.enc_len] for i in range(win_num)], 1, data_array_norm)
            return split_data
    

    def MFCLR_encoding(self, data_array_encode, test_array):
        ori_shape = data_array_encode.shape
        print("original input size : ", ori_shape)
        if self.encoding != "whole" :
            data_array_encode = data_array_encode.reshape(
                data_array_encode.shape[0] * data_array_encode.shape[1],
                data_array_encode.shape[2],
                data_array_encode.shape[3],
            )
        print("encoding input size : ", data_array_encode.shape)
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
                model.save("results/models/MFCLR/" + self.dataset + ".pkl")
        else :
            try :
                model.load("results/models/MFCLR/" + self.dataset + ".pkl")
            except :
                loss_log = model.fit(
                    train_data= data_array_encode,
                    n_epochs= self.epoch,
                    verbose= True,
                )
        
        padding = 0
        train_repr = model.encode(
            data_array_encode, 
            casual= False, 
            sliding_length= None, 
            sliding_padding= padding,
        )
        
        print("encoded train_repr.shape : ", train_repr.shape)
        if self.encoding != "whole" :
            train_repr = train_repr.reshape(ori_shape[0], ori_shape[1], train_repr.shape[1], -1)
            print("original train_repr.shape : ", train_repr.shape)
        train_repr = train_repr.reshape(train_repr.shape[0], -1)
        print("train_repr.shape : ", train_repr.shape)
        
        if self.encoding == "whole" : 
            test_repr = model.encode(test_array)
            test_repr = test_repr.reshape(test_repr.shape[0], -1)
        else : 
            test_encode = self.data_split(test_array, "cut")
            test_ori_shape = test_encode.shape
            test_encode = test_encode.reshape(
                test_encode.shape[0] * test_encode.shape[1],
                test_encode.shape[2],
                test_encode.shape[3],
            )
            test_repr = model.encode(test_encode)
            test_repr = test_repr.reshape(test_ori_shape[0], test_ori_shape[1], test_repr.shape[1], -1)
            test_repr = test_repr.reshape(test_repr.shape[0], -1)
        print("test_repr.shape : ", test_repr.shape)
        
        return train_repr, test_repr
    

    def TLoss_encoding(self, data_array_encode, test_array):
        """
        parameters meaning in  Unsupervised_Scalable:
        :params input_dims  in_channel in casula-cnn
        :params channels  channels number of channels manipulated in the casual CNN
        :params output_dims  number of output channels

        eg. input_shape (batch_size,timestep,feature)->(batch_size,feature)
        """
        ori_shape = data_array_encode.shape
        print("original input size : ", ori_shape)
        if self.encoding != "whole" :
            data_array_encode = data_array_encode.reshape(
                data_array_encode.shape[0] * data_array_encode.shape[1],
                data_array_encode.shape[2],
                data_array_encode.shape[3],
            )
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
                model.save("results/models/TLoss/" + self.dataset + ".pkl")
        else :
            try :
                model.load("results/models/TLoss/" + self.dataset + ".pkl")
            except :
                loss_log = model.fit(data_array_encode)

        train_repr = model.encode(data_array_encode)
        print("encoded train_repr.shape : ", train_repr.shape)
        if self.encoding != "whole" :
            train_repr = train_repr.reshape(ori_shape[0], ori_shape[1], train_repr.shape[1], -1)
            print("original train_repr.shape : ", train_repr.shape)
        train_repr = train_repr.reshape(train_repr.shape[0], -1)
        print("train_repr.shape : ", train_repr.shape)
        
        if self.encoding == "whole" : 
            test_repr = model.encode(test_array)
            test_repr = test_repr.reshape(test_repr.shape[0], -1)
        else : 
            test_encode = self.data_split(test_array, self.encoding)
            test_ori_shape = test_encode.shape
            test_encode = test_encode.reshape(
                test_encode.shape[0] * test_encode.shape[1],
                test_encode.shape[2],
                test_encode.shape[3],
            )
            test_repr = model.encode(test_encode)
            test_repr = test_repr.reshape(test_ori_shape[0], test_ori_shape[1], test_repr.shape[1], -1)
            test_repr = test_repr.reshape(test_repr.shape[0], -1)
        print("test_repr.shape : ", test_repr.shape)
        
        return train_repr, test_repr
    

    def TSTCC_encoding(self, data_array_encode, test_array):
        """
        parameters meaning in TSTCC_encoding:
        :params input_dims  in_channel 
        :params final_out_channels  number of output channels
        eg. input_shape (batch_size,timestep,feature)->(batch_size, final_out_channels
        """        
        ori_shape = data_array_encode.shape
        print("original input size : ", ori_shape)
        if self.encoding != "whole" :
            data_array_encode = data_array_encode.reshape(
                data_array_encode.shape[0] * data_array_encode.shape[1],
                data_array_encode.shape[2],
                data_array_encode.shape[3],
            )
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
                model.save_model("results/models/TSTCC/" + self.dataset + ".pkl")
        else :
            try :
                model.load_model("results/models/TSTCC/" + self.dataset + ".pkl")
            except :
                loss_log = model.fit(
                    data_array_encode,
                    n_epochs= self.epoch,
                )
        train_repr = model.encode(data_array_encode)
        print("encoded train_repr.shape : ", train_repr.shape)
        if self.encoding != "whole" :
            train_repr = train_repr.reshape(ori_shape[0], ori_shape[1], train_repr.shape[1], -1)
            print("original train_repr.shape : ", train_repr.shape)
        train_repr = train_repr.reshape(train_repr.shape[0], -1)
        print("train_repr.shape : ", train_repr.shape)
        
        if self.encoding == "whole" : 
            test_repr = model.encode(test_array)
            test_repr = test_repr.reshape(test_repr.shape[0], -1)
        else : 
            test_encode = self.data_split(test_array, self.encoding)
            test_ori_shape = test_encode.shape
            test_encode = test_encode.reshape(
                test_encode.shape[0] * test_encode.shape[1],
                test_encode.shape[2],
                test_encode.shape[3],
            )
            test_repr = model.encode(test_encode)
            test_repr = test_repr.reshape(test_ori_shape[0], test_ori_shape[1], test_repr.shape[1], -1)
            test_repr = test_repr.reshape(test_repr.shape[0], -1)
        print("test_repr.shape : ", test_repr.shape)
        
        return train_repr, test_repr
    

    def TNC_encoding(self, data_array_encode, test_array): 
        ori_shape = data_array_encode.shape
        print("original input size : ", ori_shape)
        if self.encoding == "whole" and ori_shape[1] < 128 :
            stretch_ratio = int(128 / ori_shape[1]) + 1
            t = np.array([data_array_encode for i in range(stretch_ratio)]).transpose(1, 2, 0, 3)
            data_array_encode = t.reshape(t.shape[0], -1, t.shape[3])
        if self.encoding != "whole" :
            data_array_encode = data_array_encode.reshape(
                data_array_encode.shape[0] * data_array_encode.shape[1],
                data_array_encode.shape[2],
                data_array_encode.shape[3],
            )
            if data_array_encode.shape[2] < 128 :
                print("Too short for TNC model. The array has to be encoded using 'whole'.")
                raise
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
                model.save("results/models/TNC/" + self.dataset + ".pkl")
        else :
            try :
                model.load("results/models/TNC/" + self.dataset + ".pkl")
            except :
                loss, accuracy = model.fit(data_array_encode, self.epoch)

        train_repr = model.encode(data_array_encode)
        print("encoded train_repr.shape : ", train_repr.shape)
        if self.encoding != "whole" :
            train_repr = train_repr.reshape(ori_shape[0], ori_shape[1], train_repr.shape[1], -1)
            print("original train_repr.shape : ", train_repr.shape)
        train_repr = train_repr.reshape(train_repr.shape[0], -1)
        print("train_repr.shape : ", train_repr.shape)
        
        if self.encoding == "whole" : 
            if self.encoding == "whole" and ori_shape[1] < 128 :
                t_test = np.array([test_array for i in range(stretch_ratio)]).transpose(1, 2, 0, 3)
                test_array = t_test.reshape(t_test.shape[0], -1, t_test.shape[3])
            test_repr = model.encode(test_array)
            test_repr = test_repr.reshape(test_repr.shape[0], -1)
        else : 
            test_encode = self.data_split(test_array)
            test_ori_shape = test_encode.shape
            test_encode = test_encode.reshape(
                test_encode.shape[0] * test_encode.shape[1],
                test_encode.shape[2],
                test_encode.shape[3],
            )
            test_repr = model.encode(test_encode)
            test_repr = test_repr.reshape(test_ori_shape[0], test_ori_shape[1], test_repr.shape[1], -1)
            test_repr = test_repr.reshape(test_repr.shape[0], -1)
        print("test_repr.shape : ", test_repr.shape)
        
        return train_repr, test_repr


    def TS2Vec_encoding(self, data_array_encode, test_array):
        ori_shape = data_array_encode.shape
        print("original input size : ", ori_shape)
        if self.encoding != "whole" :
            data_array_encode = data_array_encode.reshape(
                data_array_encode.shape[0] * data_array_encode.shape[1],
                data_array_encode.shape[2],
                data_array_encode.shape[3],
            )
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
                model.save("results/models/TS2Vec/" + self.dataset + ".pkl")
        else :
            try :
                model.load("results/models/TS2Vec/" + self.dataset + ".pkl")
            except :
                loss_log = model.fit(
                    train_data= data_array_encode,
                    n_epochs= self.epoch,
                    verbose= True
                )
        
        train_repr = model.encode(data_array_encode)
        print("encoded train_repr.shape : ", train_repr.shape)
        if self.encoding != "whole" :
            train_repr = train_repr.reshape(ori_shape[0], ori_shape[1], train_repr.shape[1], -1)
            print("original train_repr.shape : ", train_repr.shape)
        train_repr = train_repr.reshape(train_repr.shape[0], -1)
        print("train_repr.shape : ", train_repr.shape)
        
        if self.encoding == "whole" : 
            test_repr = model.encode(test_array)
            test_repr = test_repr.reshape(test_repr.shape[0], -1)
        else : 
            test_encode = self.data_split(test_array, "cut")
            test_ori_shape = test_encode.shape
            test_encode = test_encode.reshape(
                test_encode.shape[0] * test_encode.shape[1],
                test_encode.shape[2],
                test_encode.shape[3],
            )
            test_repr = model.encode(test_encode)
            test_repr = test_repr.reshape(test_ori_shape[0], test_ori_shape[1], test_repr.shape[1], -1)
            test_repr = test_repr.reshape(test_repr.shape[0], -1)
        print("test_repr.shape : ", test_repr.shape)
        
        return train_repr, test_repr
    

    def CPC_encoding(self, data_array_encode, test_array):
        ori_shape = data_array_encode.shape
        print("original input size : ", ori_shape)
        if self.encoding != "whole" :
            data_array_encode = data_array_encode.reshape(
                data_array_encode.shape[0] * data_array_encode.shape[1],
                data_array_encode.shape[2],
                data_array_encode.shape[3],
            )
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
            device='cpu',
            prediction_step= 12,
        )
        if self.training is True :
            n_epoch = self.epoch
            model.fit(data_array_encode, n_epoch)
            if self.saving is True :
                model.save_model(fn= "results/models/CPC/" + self.dataset + ".pkl")
        else :
            try :
                model.load_model("results/models/CPC/" + self.dataset + ".pkl")
            except :
                n_epoch = self.epoch
                model.fit(data_array_encode, n_epoch)

        
        train_repr = model.encode(data_array_encode)
        print("encoded train_repr.shape : ", train_repr.shape)
        if self.encoding != "whole" :
            train_repr = train_repr.reshape(ori_shape[0], ori_shape[1], train_repr.shape[1], -1)
            print("original train_repr.shape : ", train_repr.shape)
        train_repr = train_repr.reshape(train_repr.shape[0], -1)
        print("train_repr.shape : ", train_repr.shape)
        
        if self.encoding == "whole" : 
            test_repr = model.encode(test_array)
            test_repr = test_repr.reshape(test_repr.shape[0], -1)
        else : 
            test_encode = self.data_split(test_array, "cut")
            test_ori_shape = test_encode.shape
            test_encode = test_encode.reshape(
                test_encode.shape[0] * test_encode.shape[1],
                test_encode.shape[2],
                test_encode.shape[3],
            )
            test_repr = model.encode(test_encode)
            test_repr = test_repr.reshape(test_ori_shape[0], test_ori_shape[1], test_repr.shape[1], -1)
            test_repr = test_repr.reshape(test_repr.shape[0], -1)
        print("test_repr.shape : ", test_repr.shape)
        
        return train_repr, test_repr
    

    def CoST_encoding(self, data_array_encode, test_array):
        """
        value is default in model code
        """
        ori_shape = data_array_encode.shape
        print("original input size : ", ori_shape)
        if self.encoding != "whole" :
            data_array_encode = data_array_encode.reshape(
                data_array_encode.shape[0] * data_array_encode.shape[1],
                data_array_encode.shape[2],
                data_array_encode.shape[3],
            )
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
                model.save("results/models/CoST/" + self.dataset + ".pkl")
        else :
            try :
                model.load("results/models/CoST/" + self.dataset + ".pkl")
            except :
                loss_log = model.fit(
                    data_array_encode,
                    n_epochs= self.epoch,
                    verbose=True
                )
        
        train_repr = model.encode(data_array_encode)
        print("encoded train_repr.shape : ", train_repr.shape)
        if self.encoding != "whole" :
            train_repr = train_repr.reshape(ori_shape[0], ori_shape[1], train_repr.shape[1], -1)
            print("original train_repr.shape : ", train_repr.shape)
        train_repr = train_repr.reshape(train_repr.shape[0], -1)
        print("train_repr.shape : ", train_repr.shape)
        
        if self.encoding == "whole" : 
            test_repr = model.encode(test_array)
            test_repr = test_repr.reshape(test_repr.shape[0], -1)
        else : 
            test_encode = self.data_split(test_array, "cut")
            test_ori_shape = test_encode.shape
            test_encode = test_encode.reshape(
                test_encode.shape[0] * test_encode.shape[1],
                test_encode.shape[2],
                test_encode.shape[3],
            )
            test_repr = model.encode(test_encode)
            test_repr = test_repr.reshape(test_ori_shape[0], test_ori_shape[1], test_repr.shape[1], -1)
            test_repr = test_repr.reshape(test_repr.shape[0], -1)
        print("test_repr.shape : ", test_repr.shape)
        
        return train_repr, test_repr
    

    def TFC_encoding(self, data_array_encode, test_array):
        ori_shape = data_array_encode.shape
        print("original input size : ", ori_shape)
        if self.encoding == "whole" and ori_shape[1] < 128 :
            stretch_ratio = int(128 / ori_shape[1]) + 1
            t = np.array([data_array_encode for i in range(stretch_ratio)]).transpose(1, 2, 0, 3)
            data_array_encode = t.reshape(t.shape[0], -1, t.shape[3])
        if self.encoding != "whole" :
            data_array_encode = data_array_encode.reshape(
                data_array_encode.shape[0] * data_array_encode.shape[1],
                data_array_encode.shape[2],
                data_array_encode.shape[3],
            )
            if data_array_encode.shape[2] < 128 :
                print("Too short for TNC model. The array has to be encoded using 'whole'.")
                raise
        print("encoding input size : ", data_array_encode.shape)
        if data_array_encode.shape[1] % 2 == 0 : ts_aligned = data_array_encode.shape[1]
        else : ts_aligned = data_array_encode.shape[1] - 1
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
                 TSlength_aligned=ts_aligned,
        )
        if self.training is True :
            model.fit(data_array_encode, self.epoch)
            if self.saving is True :
                model.save("results/models/TFC/" + self.dataset + ".pkl")
        else :
            try :
                model.load("results/models/TFC/" + self.dataset + ".pkl")
            except :
                model.fit(data_array_encode, self.epoch)

        train_repr = model.encode(data_array_encode)
        print("encoded train_repr.shape : ", train_repr.shape)
        if self.encoding != "whole" :
            train_repr = train_repr.reshape(ori_shape[0], ori_shape[1], train_repr.shape[1], -1)
            print("original train_repr.shape : ", train_repr.shape)
        train_repr = train_repr.reshape(train_repr.shape[0], -1)
        print("train_repr.shape : ", train_repr.shape)
        
        if self.encoding == "whole" : 
            if self.encoding == "whole" and ori_shape[1] < 128 :
                t_test = np.array([test_array for i in range(stretch_ratio)]).transpose(1, 2, 0, 3)
                test_array = t_test.reshape(t_test.shape[0], -1, t_test.shape[3])
            test_repr = model.encode(test_array)
            test_repr = test_repr.reshape(test_repr.shape[0], -1)
        else : 
            test_encode = self.data_split(test_array)
            test_ori_shape = test_encode.shape
            test_encode = test_encode.reshape(
                test_encode.shape[0] * test_encode.shape[1],
                test_encode.shape[2],
                test_encode.shape[3],
            )
            test_repr = model.encode(test_encode)
            test_repr = test_repr.reshape(test_ori_shape[0], test_ori_shape[1], test_repr.shape[1], -1)
            test_repr = test_repr.reshape(test_repr.shape[0], -1)
        print("test_repr.shape : ", test_repr.shape)
        
        return train_repr, test_repr
    

    def fit_svm(self, train_features, train_y, search= False):
        min_length = min(train_features.shape[0], train_y.shape[0])
        train_features = train_features[-min_length : ]
        train_y = train_y[-min_length : ]
        print("truncated size : ", train_features.shape, train_y.shape)

        svm_clf = SVC(probability=True)
        if search is False :
            return svm_clf.fit(train_features, train_y)
        else :
            grid_search = GridSearchCV(
                svm_clf, {
                    'C': [
                        0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000,
                        np.inf
                    ],
                    'kernel': ['rbf'],
                    'degree': [3],
                    'gamma': ['scale'],
                    'coef0': [0],
                    'shrinking': [True],
                    'probability': [False],
                    'tol': [0.001],
                    'cache_size': [200],
                    'class_weight': [None],
                    'verbose': [False],
                    'max_iter': [10000000],
                    'decision_function_shape': ['ovr'],
                    'random_state': [None]
                },
                cv=5, n_jobs=-1
        )
        grid_search.fit(train_features, train_y)
        return grid_search.best_estimator_
    

    def one_hot_encoding(self, X):
        X = [int(x) for x in X]
        n_values = np.max(X) + 1
        b = np.eye(n_values)[X]
        return b
    

    def cal_loss(self, pred_y, label_y, pred_y_prob, label_y_onehot):
        acc = accuracy_score(label_y, pred_y)
        precision = precision_score(label_y, pred_y, average='macro', )
        recall = recall_score(label_y, pred_y, average='macro', )
        F1 = f1_score(label_y, pred_y, average='macro')
        auc = roc_auc_score(label_y_onehot, pred_y_prob, average="macro", multi_class="ovr")
        prc = average_precision_score(label_y_onehot, pred_y_prob, average="macro")
        return acc, precision, recall, F1, auc, prc


    def classify(self):
        train_x, train_y, test_x, test_y = UEA_resample(self.dataset)
        for i in range(train_x.shape[0]):
            if True in np.isnan(train_x[i, :, :]):
                print(i)
        print("resmapled size : ", self.dataset)
        print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

        train_x_norm, train_mean, train_std = self.normalisation(train_x)
        test_x_norm, test_mean, test_std = self.normalisation(test_x)
        print("x train shape : ", train_x.shape)
        print("x train normalised shape : ", train_x_norm.shape)
        if self.encoding == "whole" : data_array_enc = train_x_norm.copy()
        else : data_array_enc = self.data_split(train_x_norm, self.encoding)
        if self.clr_method == "T-Loss" :
            train_repr, test_repr = self.TLoss_encoding(data_array_enc, test_x_norm)
        elif self.clr_method == "TS-TCC" :
            train_repr, test_repr = self.TSTCC_encoding(data_array_enc, test_x_norm)
        elif self.clr_method == "TNC" :
            train_repr, test_repr = self.TNC_encoding(data_array_enc, test_x_norm)
        elif self.clr_method == "TS2Vec" :
            train_repr, test_repr = self.TS2Vec_encoding(data_array_enc, test_x_norm)
        elif self.clr_method == "CPC" :
            train_repr, test_repr = self.CPC_encoding(data_array_enc, test_x_norm)
        elif self.clr_method == "CoST" :
            train_repr, test_repr = self.CoST_encoding(data_array_enc, test_x_norm)
        elif self.clr_method == "MF-CLR" :
            train_repr, test_repr = self.MFCLR_encoding(data_array_enc, test_x_norm)
        elif self.clr_method == "TF-C" :
            train_repr, test_repr = self.TFC_encoding(data_array_enc, test_x_norm)
        else : raise

        clf = self.fit_svm(train_repr, train_y)
        pred_y = clf.predict(test_repr)
        pred_y_prob = clf.predict_proba(test_repr)

        test_y_onehot = self.one_hot_encoding(test_y)

        acc, precision, recall, F1, auc, prc = self.cal_loss(pred_y, test_y, pred_y_prob, test_y_onehot)
        print('classification results : ', self.clr_method, self.dataset)
        print('Acc: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f | AUROC: %.4f | AUPRC: %.4f' % (acc, precision, recall, F1, auc, prc))
        print()
        print()
        print()
        print()

        return acc, precision, recall, F1, auc, prc




if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument('--is_training', type=bool, required=True, default=1, help='status')
    parser.add_argument('--dataset', type=str, required=True, help='The dataset name, This can be set to ETTm1, ETTm2, traffic, weather')
    parser.add_argument('--method', type=str, required=True, default='mf_clr', help='model_name, options:[MF-CLR, T-Loss, TS-TCC, TNC, TS2Vec, CPC, CoST]')
    parser.add_argument('--use_gpu', type=bool, default=False, help='use_gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--batch_size', type=int, default=16, help='The batch size(defaults to 16)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate(defaluts to 0.001)')
    parser.add_argument('--epochs', type=int, default=30, help='The number of epochs(defaults to 30)')
    parser.add_argument('--enc_len', type=int, default=128, help='input sequence length')
    parser.add_argument('--ot_granu', type=str, default='quarterly', help='frequency of the forecast target')
    parser.add_argument('--mask_rate', type=str, default='0.25', help='mask ratio')
    parser.add_argument('--save', type=bool, default=True, help='save the checkpoint')
    parser.add_argument('--window', type=int, default=7, help='window')
    parser.add_argument('--split', type=float, default=0.2, help='test ratio')
    parser.add_argument('--ph_dim', type=int, default=32, help='dimension of projector')
    parser.add_argument('--out_dim', type=int, default=64, help='dimension of output')
    parser.add_argument('--hidden_dim', type=int, default=128, help='dimension of hidden')
    parser.add_argument('--depth', type=int, default=4, help='depth')
    parser.add_argument('--encoding', type=str, default='whole', help='encoding style, options:[whole, cut, slide]')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    print('Args in experiment:')
    print(args)


    class_test = Public_Dataset_Classification(args)

    acc, precision, recall, F1, auc, prc = class_test.classify()
    print()
    print()
