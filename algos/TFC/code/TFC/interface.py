import torch
import numpy as np
import torch.nn as nn
import json
import argparse
from .model import *
from .loss import *
import torch.fft as fft
from torch.utils.data import  TensorDataset, Dataset
from .augmentations import DataTransform_FD, DataTransform_TD




class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, target_dataset_size=64, subset=False):
        super(Load_Dataset, self).__init__()
        # self.training_mode = training_mode
        # X_train = dataset["samples"]
        # y_train = dataset["labels"]
        # shuffle
        # data = list(zip(X_train, y_train))
        np.random.shuffle(dataset)
        X_train = torch.from_numpy(dataset)

        # dataset = TensorDataset(torch.from_numpy(dataset).to(torch.float))

        # X_train, y_train = zip(*data)
        # X_train = torch.stack(list(dataset), dim=0)

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        """Align the TS length between source and target datasets"""
        X_train = X_train[:, :1, :int(config.TSlength_aligned)] # take the first 178 samples

        """Subset for debugging"""
        if subset == True:
            subset_size = target_dataset_size * 10 #30 #7 # 60*1
            """if the dimension is larger than 178, take the first 178 dimensions. If multiple channels, take the first channel"""
            X_train = X_train[:subset_size]
            # y_train = y_train[:subset_size]
            # print('Using subset for debugging, the datasize is:', y_train.shape[0])

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            # self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            # self.y_data = y_train

        """Transfer x_data to Frequency Domain. If use fft.fft, the output has the same shape; if use fft.rfft, 
        the output shape is half of the time window."""

        window_length = self.x_data.shape[-1]
        self.x_data_f = fft.fft(self.x_data).abs() #/(window_length) # rfft for real value inputs.
        self.len = X_train.shape[0]

        """Augmentation"""
        # if training_mode == "pre_train":  # no need to apply Augmentations in other modes
        self.aug1 = DataTransform_TD(self.x_data, config)
        self.aug1_f = DataTransform_FD(self.x_data_f, config) # [7360, 1, 90]

    def __getitem__(self, index):
        # if self.training_mode == "pre_train":
        return self.x_data[index], self.aug1[index],  self.x_data_f[index], self.aug1_f[index]
        # else:
        #     return self.x_data[index], self.y_data[index], self.x_data[index], \
        #            self.x_data_f[index], self.x_data_f[index]

    def __len__(self):
        return self.len




def data_generator(train_dataset, configs,  subset=False):
    # train_dataset = torch.load(os.path.join(sourcedata_path, "train.pt"))
    # finetune_dataset = torch.load(os.path.join(targetdata_path, "train.pt"))  # train.pt
    # test_dataset = torch.load(os.path.join(targetdata_path, "test.pt"))  # test.pt
    """In pre-training: 
    train_dataset: [371055, 1, 178] from SleepEEG.    
    finetune_dataset: [60, 1, 178], test_dataset: [11420, 1, 178] from Epilepsy"""

    # subset = True # if true, use a subset for debugging.
    train_dataset = Load_Dataset(train_dataset, configs, target_dataset_size=configs.batch_size, subset=subset) # for self-supervised, the data are augmented here
    if len(train_dataset) < configs.batch_size:
        batch_size = 16
    else:
        batch_size = configs.batch_size
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=True,
                                               num_workers=0)
    return train_loader



class TFC_inter:
    def __init__(self,
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
                 TSlength_aligned = 128
                 ):
            super().__init__()
            self.device = device
            augmentation = argparse.Namespace(**{
                "jitter_scale_ratio": 1.1,
                "jitter_ratio": 0.8,
                "max_seg": 8
            })
            TC = argparse.Namespace(**{
                "hidden_dim": 100,
                "timesteps": 6
            })
            Context_Cont= argparse.Namespace(**{
                "temperature": 0.2,
                "use_cosine_similarity": True
            })
            config = {
                "lr": lr,
                "input_channels": input_channels,
                "kernal_size": kernel_size,
                "stride": stride,
                "final_out_channels": final_out_channels,
                "dropout": dropout,
                "features_len": features_len,
                "beta1": beta1,
                "beta2": beta2,
                "lr": lr,
                "drop_last": True,
                "batch_size": batch_size,
                "Context_Cont":Context_Cont,
                "TC": TC,
                "augmentation": augmentation,
                "TSlength_aligned": TSlength_aligned,
                "lr_f": lr,
                "target_batch_size": 42,
                "increased_dim": 1,
                "features_len_f": features_len,
                "CNNoutput_channel": 28
            }
            self.configs = argparse.Namespace(**config)
            self.TFC_model = TFC(self.configs).to(self.device)
            self.model_optimizer = torch.optim.Adam(self.TFC_model.parameters(), lr=self.configs.lr, betas=(self.configs.beta1, self.configs.beta2), weight_decay=3e-4)

            
    def fit(self, train_dataset, n_epochs):
        train_dl= data_generator(train_dataset, self.configs)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.model_optimizer, 'min')
        for epoch in range(1, n_epochs + 1):
            # Train and validate
            """Train. In fine-tuning, this part is also trained???"""
            train_loss = self.model_pretrain(train_dl, self.configs, self.device)
            print(f'\nPre-training Epoch : {epoch}', f'Train Loss : {train_loss:.4f}')

  

    def model_pretrain(self, train_loader, config, device):
        total_loss = []
        self.TFC_model.train()
        global loss, loss_t, loss_f, l_TF, loss_c, data_test, data_f_test
        # optimizer
        self.model_optimizer.zero_grad()
        for batch_idx, (data, aug1, data_f, aug1_f) in enumerate(train_loader):
            data = data.float().to(device) # data: [128, 1, 178], labels: [128]
            aug1 = aug1.float().to(device)  # aug1 = aug2 : [128, 1, 178]
            data_f, aug1_f = data_f.float().to(device), aug1_f.float().to(device)  # aug1 = aug2 : [128, 1, 178]
            """Produce embeddings"""
            h_t, z_t, h_f, z_f = self.TFC_model(data, data_f)
            h_t_aug, z_t_aug, h_f_aug, z_f_aug = self.TFC_model(aug1, aug1_f)
            """Compute Pre-train loss"""
            """NTXentLoss: normalized temperature-scaled cross entropy loss. From SimCLR"""
            nt_xent_criterion = NTXentLoss_poly(device, config.batch_size, config.Context_Cont.temperature,
                                        config.Context_Cont.use_cosine_similarity) # device, 128, 0.2, True
            loss_t = nt_xent_criterion(h_t, h_t_aug)
            loss_f = nt_xent_criterion(h_f, h_f_aug)
            l_TF = nt_xent_criterion(z_t, z_f) # this is the initial version of TF loss
            l_1, l_2, l_3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(z_t_aug, z_f), nt_xent_criterion(z_t_aug, z_f_aug)
            loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)
            lam = 0.2
            loss = lam*(loss_t + loss_f) + l_TF
            total_loss.append(loss.item())
            loss.backward()
            self.model_optimizer.step()
        print('Pretraining: overall loss:{}, l_t: {}, l_f:{}, l_c:{}'.format(loss, loss_t, loss_f, l_TF))
        ave_loss = torch.tensor(total_loss).mean()
        return ave_loss


    def encode(self, dataset):
        self.TFC_model.eval()
        dl= data_generator(dataset, self.configs)
        dataset = Load_Dataset(dataset, self.configs, target_dataset_size=self.configs.batch_size, subset=False) # for self-supervised, the data are augmented here
        dl = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.configs.batch_size)
        print("dl length", len(dl))
        output = []
        with torch.no_grad():
            for data, _, data_f, _ in dl:
                data = data.float().to(self.device)
                data_f = data_f.float().to(self.device)
 
                """Add supervised classifier: 1) it's unique to finetuning. 2) this classifier will also be used in test"""
                h_t, z_t, h_f, z_f = self.TFC_model(data, data_f)
                fea_concat = torch.cat((z_t, z_f), dim=1)
                output.append(fea_concat)
            print(len(output))
            output = torch.cat(output, dim=0)
        return output.numpy()


    
    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        chkpoint = {'model_state_dict': self.TFC_model.state_dict()}
        torch.save(chkpoint, fn)
        print('Pretrained model is stored at folder:{}'.format(fn))


    def load(self, fn):
        chkpoint = torch.load(fn, map_location=self.device)
        self.TFC_model.load_state_dict(chkpoint)
