import torch
import math
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from statsmodels.tsa.stattools import adfuller
from .tnc.models import WFEncoder, Discriminator


class TNCDataset(data.Dataset):
    def __init__(self, x, mc_sample_size, window_size, augmentation, epsilon=3, state=None, adf=False):
        super(TNCDataset, self).__init__()
        self.time_series = x
        self.T = x.shape[-1]
        self.window_size = window_size
        self.sliding_gap = int(window_size*25.2)
        self.window_per_sample = (self.T-2*self.window_size)//self.sliding_gap
        self.mc_sample_size = mc_sample_size
        self.state = state
        self.augmentation = augmentation
        self.adf = adf
        if not self.adf:
            self.epsilon = epsilon
            self.delta = 5*window_size*epsilon

    def __len__(self):
        return len(self.time_series)*self.augmentation

    def __getitem__(self, ind):
        ind = ind%len(self.time_series)

        t = np.random.randint(2*self.window_size, self.T-2*self.window_size)
        x_t = self.time_series[ind][:,t-self.window_size//2:t+self.window_size//2]
        X_close = self._find_neighours(self.time_series[ind], t)
        X_distant = self._find_non_neighours(self.time_series[ind], t)

        if self.state is None:
            y_t = -1
        else:
            y_t = torch.round(torch.mean(self.state[ind][t-self.window_size//2:t+self.window_size//2]))
        return x_t, X_close, X_distant, y_t

    def _find_neighours(self, x, t):
        T = self.time_series.shape[-1]
        if self.adf:
            gap = self.window_size
            corr = []
            for w_t in range(self.window_size,4*self.window_size, gap):
                try:
                    p_val = 0
                    for f in range(x.shape[-2]):
                        p = adfuller(np.array(x[f, max(0,t - w_t):min(x.shape[-1], t + w_t)].reshape(-1, )))[1]
                        p_val += 0.01 if math.isnan(p) else p
                    corr.append(p_val/x.shape[-2])
                except:
                    corr.append(0.6)
            self.epsilon = len(corr) if len(np.where(np.array(corr) >= 0.01)[0])==0 else (np.where(np.array(corr) >= 0.01)[0][0] + 1)
            self.delta = 5*self.epsilon*self.window_size

        ## Random from a Gaussian
        t_p = [int(t+np.random.randn()*self.epsilon*self.window_size) for _ in range(self.mc_sample_size)]
        t_p = [max(self.window_size//2+1,min(t_pp,T-self.window_size//2)) for t_pp in t_p]
        x_p = torch.stack([x[:, t_ind-self.window_size//2:t_ind+self.window_size//2] for t_ind in t_p])
        return x_p

    def _find_non_neighours(self, x, t):
        T = self.time_series.shape[-1]
        if t>T/2:
            t_n = np.random.randint(self.window_size//2, max((t - self.delta + 1), self.window_size//2+1), self.mc_sample_size)
        else:
            t_n = np.random.randint(min((t + self.delta), (T - self.window_size-1)), (T - self.window_size//2), self.mc_sample_size)
        x_n = torch.stack([x[:, t_ind-self.window_size//2:t_ind+self.window_size//2] for t_ind in t_n])

        if len(x_n)==0:
            rand_t = np.random.randint(0,self.window_size//5)
            if t > T / 2:
                x_n = x[:,rand_t:rand_t+self.window_size].unsqueeze(0)
            else:
                x_n = x[:, T - rand_t - self.window_size:T - rand_t].unsqueeze(0)
        return x_n



class TNC:
    def __init__(self,
                 encoding_size=128,
                 device='cpu',
                 batch_size=16,
                 lr=0.001,
                 cv = 1,
                 w = 20,
                 decay=0.005,
                 mc_sample_size=20,
                 augmentation=1,
                 channels=7
                 ):
            super().__init__()
            self.lr = lr
            self.cv = cv
            self.w = w
            self.mc_sample_size = mc_sample_size
            self.decay = decay
            self.device = device
            self.augmentation = augmentation
            self.batch_size = batch_size
            self.encoding_size = encoding_size
            self.channels = channels
            self.model = WFEncoder(encoding_size=self.encoding_size, channels=self.channels).to(self.device)
            self.disc_model = Discriminator(self.encoding_size, self.device)
            

    def fit(self, train_dataset, n_epochs):
        accuracies, losses = [], []
        train_dataset = torch.from_numpy(train_dataset)
        train_dataset = train_dataset.permute(0, 2, 1).numpy()
        T = train_dataset.shape[-1]
        # x_window = np.concatenate(np.split(train_dataset[:, :, :T // 5 * 5], 5, -1), 0)

        params = list(self.disc_model.parameters()) + list(self.model.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.decay)
        inds = list(range(len(train_dataset)))
        random.shuffle(inds)
        train_dataset = train_dataset[inds]
        n_train = int(0.8*len(train_dataset))
        performance = []
        best_acc = 0
        best_loss = np.inf
        for epoch in range(n_epochs+1):
            print("Epoch", epoch)
            print("Preparing trainset")
            trainset = TNCDataset(x=torch.Tensor(train_dataset[:n_train]), mc_sample_size=self.mc_sample_size,
                                  window_size=self.w, augmentation=self.augmentation, adf=True)
            print("Preparing trainloader")
            train_loader = data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True) # why use num_workers=3???
            print("Preparing validset")
            validset = TNCDataset(x=torch.Tensor(train_dataset[n_train:]), mc_sample_size=self.mc_sample_size,
                                  window_size=self.w, augmentation=self.augmentation, adf=True)
            print("Preparing validloader")
            valid_loader = data.DataLoader(validset, batch_size=self.batch_size, shuffle=True)
            print("Running epoch")
            epoch_loss, epoch_acc = self.epoch_run(train_loader, self.disc_model, self.model, optimizer=optimizer,
                                              w=self.w, train=True, device=self.device)
            test_loss, test_acc = self.epoch_run(valid_loader, self.disc_model, self.model, train=False, w=self.w, device=self.device)
            print("Do some other stuff")
            performance.append((epoch_loss, test_loss, epoch_acc, test_acc))
            if epoch%10 == 0:
                print('(cv:%s)Epoch %d Loss =====> Training Loss: %.5f \t Training Accuracy: %.5f \t Test Loss: %.5f \t Test Accuracy: %.5f'
                      % (self.cv, epoch, epoch_loss, epoch_acc, test_loss, test_acc))
            if best_loss > test_loss:
                best_acc = test_acc
                best_loss = test_loss
                state = {
                    'epoch': epoch,
                    'encoder_state_dict': self.model.state_dict(),
                    'discriminator_state_dict': self.disc_model.state_dict(),
                    'best_accuracy': test_acc
                }
                torch.save(state, './checkpoint_%d.pth.tar'%(self.cv)) # passed thru on first run
            print("Epoch ends")
        accuracies.append(best_acc)
        losses.append(best_loss)
        # Save performance plots
        train_loss = [t[0] for t in performance]
        test_loss = [t[1] for t in performance]
        train_acc = [t[2] for t in performance]
        test_acc = [t[3] for t in performance]

        return losses, accuracies


    def epoch_run(self, loader, disc_model, encoder, device, w=0, optimizer=None, train=True):
        if train:
            encoder.train()
            disc_model.train()
        else:
            encoder.eval()
            disc_model.eval()
        # loss_fn = torch.nn.BCELoss()
        loss_fn = torch.nn.BCEWithLogitsLoss()
        encoder.to(device)
        disc_model.to(device)
        epoch_loss = 0
        epoch_acc = 0
        batch_count = 0
        for x_t, x_p, x_n, _ in loader:
            mc_sample = x_p.shape[1]
            batch_size, f_size, len_size = x_t.shape
            x_p = x_p.reshape((-1, f_size, len_size))
            x_n = x_n.reshape((-1, f_size, len_size))
            x_t = np.repeat(x_t, mc_sample, axis=0)
            neighbors = torch.ones((len(x_p))).to(device)
            non_neighbors = torch.zeros((len(x_n))).to(device)

            x_t = x_t.to(device)
            x_p = x_p.to(device)
            x_n = x_n.to(device)

            z_t = encoder(x_t)
            z_p = encoder(x_p)
            z_n = encoder(x_n)

            d_p = disc_model(z_t, z_p)
            d_n = disc_model(z_t, z_n)

            p_loss = loss_fn(d_p, neighbors)
            n_loss = loss_fn(d_n, non_neighbors)
            n_loss_u = loss_fn(d_n, neighbors)
            loss = (p_loss + w*n_loss_u + (1-w)*n_loss)/2

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            p_acc = torch.sum(torch.nn.Sigmoid()(d_p) > 0.5).item() / len(z_p)
            n_acc = torch.sum(torch.nn.Sigmoid()(d_n) < 0.5).item() / len(z_n)
            epoch_acc = epoch_acc + (p_acc+n_acc)/2
            epoch_loss += loss.item()
            batch_count += 1
        return epoch_loss/batch_count, epoch_acc/batch_count


    def encode(self,dataset):
        self.model.eval()
        dataset = torch.from_numpy(dataset)
        dataset = dataset.permute(0, 2, 1).numpy()
        dataset = TNCDataset(x=torch.Tensor(dataset), mc_sample_size=self.mc_sample_size,
                                  window_size=self.w, augmentation=self.augmentation, adf=False)
        train_loader = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True) # why use num_workers=3???

        output = []

        with torch.no_grad():
            for id, (x_t, x_p, x_n, _)  in enumerate(train_loader):
                x_t = x_t.float().to(self.device)
                out = self.model(x_t)
                output.append(out)    
            output = torch.cat(output, dim=0)
        return output.numpy()


    def save(self, fn):
        torch.save(self.model.state_dict(), fn)
    

    def load(self, fn):
        state_dict = torch.load(fn, map_location=self.device)
        self.model.load_state_dict(state_dict)