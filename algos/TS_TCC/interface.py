import torch
import numpy as np
import torch.nn as nn
from .models.TC import TC
import torch.nn.functional as F
from .models.loss import NTXentLoss
from .models.model import base_Model
from torch.utils.data import Dataset
from .dataloader.augmentations import DataTransform
from .config_files.interface_Configs import Config

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, training_mode= "self_supervised"):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        X_train = torch.from_numpy(dataset)

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)
        # if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
        X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
        else:
            self.x_data = X_train


        self.len = X_train.shape[0]
        if training_mode == "self_supervised":  # no need to apply Augmentations in other modes
            self.aug1, self.aug2 = DataTransform(self.x_data, config)

    def __getitem__(self, index):
        if self.training_mode == "self_supervised":
            return self.x_data[index], self.aug1[index], self.aug2[index]
        else:
            return self.x_data[index], self.x_data[index], self.x_data[index]

    def __len__(self):
        return self.len

class TS_TCC:
    def __init__(self,
                    input_dims: int,
                    kernel_size: int,
                    stride: int,
                    final_out_channels: int,
                    tc_timestep: int=6,
                    device: 'str' ='cuda',
                    lr: float = 0.001,
                    batch_size: int = 16,
                     ):

            super().__init__()
            self.device = device
            self.lr = lr
            self.batch_size = batch_size
            self.configs = Config()
            self.configs.kernel_size = kernel_size
            self.configs.stride = stride
            self.configs.final_out_channels = final_out_channels
            self.configs.input_channels = input_dims
            self.configs.TC.timesteps = tc_timestep
            self.model = base_Model(self.configs).to(device)
            self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.configs.lr, betas=(self.configs.beta1, self.configs.beta2), weight_decay=3e-4)
            self.temporal_contr_model = TC(self.configs, device).to(device)
            self.temporal_contr_optimizer = torch.optim.Adam(self.temporal_contr_model.parameters(), lr=self.configs.lr, betas=(self.configs.beta1, self.configs.beta2), weight_decay=3e-4)
            
            self.n_epochs = 0
            self.n_iters = 0

    def fit(self, train_dataset, n_epochs):
        training_mode = "self_supervised"
        train_dataset = Load_Dataset(train_dataset, self.configs)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size,
                                                shuffle=True, drop_last=self.configs.drop_last,
                                                num_workers=0)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.model_optimizer, 'min')
        
        for epoch in range(1, n_epochs + 1):
            # Train and validate
            train_loss = self.model_train(self.model_optimizer, self.temporal_contr_optimizer, criterion, train_loader, self.configs, self.device, training_mode)
            # valid_loss, valid_acc, _, _ = model_evaluate(model, temporal_contr_model, valid_dl, device, training_mode)


            print(f'\nEpoch : {epoch}\n'
                    f'Train Loss     : {train_loss:.4f}\t')

                        # f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n')
                        #  f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')


    
    
    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        chkpoint = {'model_state_dict': self.model.state_dict(), 'temporal_contr_model_state_dict': self.temporal_contr_model.state_dict()}
        torch.save(chkpoint, fn)
    

    def save_model(self, fn):
        param_dict = {
            'model_state_dict': self.model.state_dict(), 
            'temporal_contr_model_state_dict': self.temporal_contr_model.state_dict()
        }
        torch.save(param_dict, fn)
    

    def load_model(self, fn):
        param_dict = torch.load(fn)
        self.model.load_state_dict(param_dict['model_state_dict'])
        self.temporal_contr_model.load_state_dict(param_dict['temporal_contr_model_state_dict'])


    def model_train(self, model_optimizer, temp_cont_optimizer, criterion, train_loader, config, device, training_mode):
        total_loss = []
        self.model.train()
        self.temporal_contr_model.train()
        for batch_idx, (data, aug1, aug2) in enumerate(train_loader):
            # send to device
            data = data.float().to(self.device)
            #（b, feature, time_stamp）
            aug1, aug2 = aug1.float().to(self.device), aug2.float().to(self.device)
            # optimizer
            model_optimizer.zero_grad()
            temp_cont_optimizer.zero_grad()

            if training_mode == "self_supervised":
                features1 = self.model(aug1)
                features2 = self.model(aug2)
                # normalize projection feature vectors
                features1 = F.normalize(features1, dim=1)
                features2 = F.normalize(features2, dim=1)
                temp_cont_loss1, temp_cont_lstm_feat1 = self.temporal_contr_model(features1, features2)
                temp_cont_loss2, temp_cont_lstm_feat2 = self.temporal_contr_model(features2, features1)

                # normalize projection feature vectors
                zis = temp_cont_lstm_feat1 
                zjs = temp_cont_lstm_feat2 

            else:
                pass

            # compute loss
            if training_mode == "self_supervised":
                lambda1 = 1
                lambda2 = 0.7
                nt_xent_criterion = NTXentLoss(device, self.batch_size, config.Context_Cont.temperature,
                                            config.Context_Cont.use_cosine_similarity)
                loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 +  nt_xent_criterion(zis, zjs) * lambda2
                
            else: # supervised training or fine tuining
                pass

            total_loss.append(loss.item())
            loss.backward()
            model_optimizer.step()
            temp_cont_optimizer.step()

        total_loss = torch.tensor(total_loss).mean()

        return total_loss


    def encode(self,dataset):
        self.model.eval()
        self.temporal_contr_model.eval()
        train_dataset = Load_Dataset(dataset, self.configs)
        loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size)

        output = []

        with torch.no_grad():
            for id, (data,_,_) in enumerate(loader):
                data = data.float().to(self.device)
                out = self.model(data)
                output.append(out)    
            output = torch.cat(output, dim=0)
        return output.numpy()

