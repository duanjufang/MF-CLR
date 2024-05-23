import torch
import argparse
from torch.utils import data
from .modules.audio.model import Model




class Contrasive:
    def __init__(self,
                 strides = [2, 2],
                 filter_sizes = [2, 2],
                 padding = [2, 1],
                 genc_input = 7,
                 genc_hidden = 512,
                 gar_hidden = 256,
                 lr = 2.0e-4,
                 batch_size = 16,
                 prediction_step = 12,
                 device='cpu'
                 ):
            super().__init__()
            self.lr = lr
            self.device = device
            self.stides = strides
            self.batch_size = batch_size
            self.filter_sizes = filter_sizes
            self.padding = padding
            self.genc_input = genc_input
            self.genc_hidden = genc_hidden
            self.gar_hidden = gar_hidden 
            config = {
                "learning_rate": self.lr,
                "prediction_step": prediction_step, # Time steps k to predict into future
                "negative_samples": 10, # Number of negative samples to be used for training
                "subsample":True,
                "batch_size": self.batch_size,
                "calc_accuracy": False
            }
            self.args = argparse.Namespace(**config)
            self.model = Model(
                    self.args,
                    strides,
                    filter_sizes,
                    padding,
                    self.genc_input,
                    genc_hidden,
                    gar_hidden,)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, train_dataset, n_epochs):
        self.model = self.model.to(self.device)
        train_dataset = torch.from_numpy(train_dataset)
        train_dataset = train_dataset.permute(0, 2, 1)
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)       
        print_idx = 100
        best_loss = 0
        global_step = 0
        for epoch in range(n_epochs):
            loss_epoch = 0
            for step, batch in enumerate(train_loader):
                # if step % validation_idx == 0:
                      # 这一步是拿feature和label通过tsne去观测效果，无label用不到
                #     validate_speakers(args, train_dataset, self.model, self.optimizer, epoch, step, global_step)
                batch = batch.to(self.device)
                # forward
                loss = self.model(batch)
                # accumulate losses for all GPUs
                loss = loss.mean()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                # backward, depending on mixed-precision
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                if step % print_idx == 0:
                    print(
                        "[Epoch {}/{}] Train step {:04d}/{:04d} \t "
                        "Loss = {:.4f} \t".format(
                            epoch,
                            n_epochs,
                            step,
                            len(train_loader),
                            loss,
                        )
                    )
                loss_epoch += loss
                global_step += 1
            avg_loss = loss_epoch / len(train_loader)
            if avg_loss > best_loss:
                best_loss = avg_loss
                # self.save_model()
            # save current model state
            # self.save_model()

        
    def save_model(self, fn):
        torch.save(self.model.state_dict(), fn)
    

    def load_model(self, fn):
        state_dict = torch.load(fn, map_location=self.device)
        self.model.load_state_dict(state_dict)
    

    def encode(self,dataset):
        self.model.eval()
        dataset = torch.from_numpy(dataset)
        dataset = dataset.permute(0, 2, 1)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)

        output = []

        with torch.no_grad():
            for id, data in enumerate(loader):
                data = data.float().to(self.device)
                z, out = self.model.model.get_latent_representations(data)
                output.append(out)    
            output = torch.cat(output, dim=0)
        return output.numpy()

