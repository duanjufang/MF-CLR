import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from . import scikit_wrappers




class Unsupervided_Scalable(nn.Module):
    def __init__(self,
                    input_dims: int,
                    device: 'str' ='cuda',
                    lr: float = 0.001,
                    batch_size: int = 16,
                    output_dims: int=320,
                    channels: int=40,
                    compared_length=None,
                    depth=10,
                    kernel_size=3,
                    penalty=None,
                    early_stopping=None,
                    nb_random_samples=10,
                    negative_penalty=1,
                    reduced_size = 160,
                    nb_steps=10
                     ):

            super().__init__()
            
            self.device = device
            self.params = {
                "batch_size": batch_size,
                "channels": channels,
                "compared_length": compared_length,
                "depth": depth,
                "nb_steps": nb_steps,
                "in_channels": input_dims,
                "kernel_size": kernel_size,
                "penalty": penalty,
                "early_stopping": early_stopping,
                "lr": lr,
                "nb_random_samples": nb_random_samples,
                "negative_penalty": negative_penalty,
                "out_channels": output_dims,
                "reduced_size": reduced_size,
                "cuda":False,
                "gpu":0
            }
            self.model = scikit_wrappers.CausalCNNEncoderClassifier()



    def fit(self, train_dataset):
        self.model.set_params(**self.params)
        dataset = torch.from_numpy(train_dataset)
        dataset = dataset.permute(0, 2, 1).numpy()
        self.model.fit(dataset, None, save_memory=True, verbose=True)



    def encode(self,dataset):
        dataset = torch.from_numpy(dataset)
        dataset = dataset.permute(0, 2, 1).numpy()
        features = self.model.encode(dataset)
        features = np.expand_dims(features, 1)
        return features


    def save(self, fn):
        # torch.save(self.model.state_dict(), fn)
        self.model.save(fn)
    

    def load(self, fn):
        # state_dict = torch.load(fn, map_location=self.device)
        # self.model.load_state_dict(state_dict)
        self.model.load(fn)