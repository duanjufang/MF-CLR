import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from time import time
from encoder import BackboneEncoder, Projector
from lossfunc import combine_loss
from util_func import take_per_row_multigrain, split_with_nan, centerize_vary_length_series, torch_pad_nan
from data_augmentation import data_generation_multi




class MF_CLR:
    
    def __init__(self, input_dims, grain_split, total_dim, output_dims= None, hidden_dims= 256, depth= 7, 
                 device= 'cpu', lr= 0.001, batch_size= 64, max_train_length= None, temporal_unit= 0, 
                 ph_dim = 128, after_iter_callback= None, after_epoch_callback= None, projection= True, da= "proposed"):
        ''' Initialize a MF_CLR model.
        Args:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims (list): The representation dimension on each granularity.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            lr (int): The learning rate.
            batch_size (int): The batch size.
            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
            temporal_unit (int): The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
            after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
            after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.
        '''
        super().__init__()
        self.device = device
        if output_dims is not None :
            self.output_dim_list = [output_dims for i in range(len(grain_split) - 1)]
        else :
            self.output_dim_list = [320 for i in range(len(grain_split) - 1)]
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit
        self.grain_split_list = grain_split      
        
        assert len(self.output_dim_list) == len(self.grain_split_list) - 1 
        self._net_list, self.net_list = [], []
        for l in range(len(grain_split) - 1):
            if len(self.net_list) == 0 :
                input_dim_l = input_dims
                output_dim_l = self.output_dim_list[0]
            else :
                if projection is False :
                    input_dim_l = self.output_dim_list[l - 1]
                    output_dim_l = self.output_dim_list[l]
                else :
                    input_dim_l = self.output_dim_list[l - 1] + ph_dim * l
                    output_dim_l = self.output_dim_list[l] + ph_dim * l
            _net = BackboneEncoder(
                input_dims= input_dim_l, 
                output_dims= output_dim_l, 
                hidden_dims= hidden_dims, 
                depth= depth,
            ).to(self.device)            
            net = torch.optim.swa_utils.AveragedModel(_net)
            net.update_parameters(_net)
            self._net_list.append(_net)
            self.net_list.append(net)
        
        self.ph_list = []
        for l in range(len(grain_split) - 1):
            projector = Projector(
                input_dims= grain_split[l + 1] - grain_split[l],
                h1= int(ph_dim / 2),
                h2= ph_dim,
            ).to(self.device)            
            self.ph_list.append(projector)

        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        self.projection = projection

        assert da in ["proposed", "scaling", "shifting", "jittering", "permutation", "random mask"]
        self.da_method = da

        self.n_epochs = 0
        self.n_iters = 0
    



    def fit(self, train_data, n_epochs=None, n_iters=None, verbose=False):
        ''' Training the MF-CLR model.
        
        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            loss_log: a list containing the training losses on each epoch.
        ''' 
        assert train_data.ndim == 3
        

        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600  # default param for n_iters
        

        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)


        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)

        if temporal_missing[0] or temporal_missing[-1]:    
            train_data = centerize_vary_length_series(train_data)

        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]  
        

        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
        train_loader = DataLoader(train_dataset, batch_size= min(self.batch_size, len(train_dataset)), shuffle= False, drop_last= True)
        optimizer_list, scheduler_list = [], []
        for g in range(len(self.grain_split_list) - 1):
            optimizer = torch.optim.AdamW(list(self._net_list[g].parameters()) + list(self.ph_list[g].parameters()), lr= self.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
            optimizer_list.append(optimizer)
            scheduler_list.append(scheduler)


        loss_log = []
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs :
                break
            
            cum_loss = 0
            n_epoch_iters = 0
            
            interrupted = False
            for batch in train_loader :
                if n_iters is not None and self.n_iters >= n_iters :
                    interrupted = True
                    break
                
                x = batch[0]    
                if self.max_train_length is not None and x.size(1) > self.max_train_length :
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[ : , window_offset : window_offset + self.max_train_length]
                x = x.to(self.device)

                ts_l = x.size(1)      
                grain_data_list = take_per_row_multigrain(x, np.random.randint(low= 0, high= 1, size= x.size(0)), ts_l, self.grain_split_list)


                for g in range(len(self.grain_split_list) - 1):
                    
                    if g == 0 :
                        fine_grained = grain_data_list[0]
                        coarse_grained = grain_data_list[1]
                    else :
                        fine_grained = out_embed
                        coarse_grained = grain_data_list[g + 1]


                    this_optimiser = optimizer_list[g]
                    this_optimiser.zero_grad()

                    fine_grained = fine_grained.to(self.device)


    
                    out = self._net_list[g](fine_grained)
                    time_start = time()
                    fine_grained_aug = torch.from_numpy(data_generation_multi(fine_grained.cpu().numpy())).float()
                    fine_grained_aug = fine_grained_aug.to(self.device)
                    out_aug = self._net_list[g](fine_grained_aug)
                    coarse_grained = coarse_grained.to(self.device)

                    grain_loss = combine_loss(out, out_aug, coarse_grained, coarse_grained)
                    grain_loss.backward()

                    cum_loss += grain_loss.item()


                    fine_embed = out.clone().detach().cpu().numpy()
                    if self.projection is True :
                        coarse_embed = self.ph_list[g](coarse_grained)
                        coarse_embed = coarse_embed.clone().detach().cpu().numpy()
                        out_embed = np.concatenate([fine_embed, coarse_embed], axis= 2)
                    else :
                        out_embed = np.concatenate([fine_embed, coarse_embed], axis= 2)
                    out_embed = torch.from_numpy(out_embed)


                    this_optimiser.step()
                    self.net_list[g].update_parameters(self._net_list[g])

                    
                    n_epoch_iters += 1
                    self.n_iters += 1
                    
                    if self.after_iter_callback is not None:
                        self.after_iter_callback(self, grain_loss.item())

            
            if interrupted:
                break
            
            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            if verbose :
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1
            
            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)
            
            for each_scheduler in scheduler_list :
                each_scheduler.step()
                
            
        return loss_log
    



    def _eval_with_pooling(self, x, encoding_net, mask= None, slicing= None, encoding_window= None):
        
        out = encoding_net(x.to(self.device, non_blocking= True), mask)
        
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[ : , slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = out.size(1),
            ).transpose(1, 2)
            
        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = encoding_window,
                stride = 1,
                padding = encoding_window // 2
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[ : , : -1]
            if slicing is not None:
                out = out[ : , slicing]
            
        elif encoding_window == 'multiscale':
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):  
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size = (1 << (p + 1)) + 1,
                    stride = 1,
                    padding = 1 << p
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)
            
        else:
            if slicing is not None:
                out = out[:, slicing]
            
        return out.cpu()
    



    def encode(self, data, mask=False, encoding_window=None, casual=False, sliding_length=None, sliding_padding=0, batch_size=None):
        ''' Compute representations using the model.
        
        Args:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            casual (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
            
        Returns:
            repr: The representations for data.
        '''

        for each_net in self.net_list :
            assert each_net is not None, 'please train or load a net first'
        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training_list = []
        for each_net in self.net_list :
            org_training_list.append(each_net.training)
            each_net.eval()
        
        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                if sliding_length is not None:
                    reprs = []
                    for i in range(0, ts_l, sliding_length):  
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not casual else 0)
                        x_sliding = torch_pad_nan(          
                            x[ : , max(l, 0) : min(r, ts_l)],
                            left= -l if l < 0 else 0,
                            right= r - ts_l if r > ts_l else 0,
                            dim= 1
                        )
                        for g in range(len(self.grain_split_list) - 1):
                            if g == 0 :
                                encoding_x_sliding = x_sliding[ : , : , : self.grain_split_list[0]]
                                encoding_f_sliding = x_sliding[ : , : , self.grain_split_list[0] : self.grain_split_list[1]]
                                encoding_net = self.net_list[0]
                                out_g = self._eval_with_pooling(
                                    encoding_x_sliding,
                                    encoding_net,
                                    slicing= slice(sliding_padding, sliding_padding + sliding_length),
                                    encoding_window= encoding_window
                                )
                                ph_net = self.ph_list[0]
                                out_f = ph_net(encoding_f_sliding)
                            else :
                                encoding_x_sliding = out_g
                                encoding_f_sliding = out_f
                                encoding_x_sliding = torch.from_numpy(np.concatenate([encoding_x_sliding.numpy(), encoding_f_sliding.numpy()], axis= 2))
                                encoding_net = self.net_list[g]
                                out_g = self._eval_with_pooling(
                                    encoding_x_sliding,
                                    encoding_net,
                                    slicing= slice(sliding_padding, sliding_padding + sliding_length),
                                    encoding_window= encoding_window
                                )
                        reprs.append(out_g)
                            
                    
                    out = torch.cat(reprs, dim=1)       
                    if encoding_window == 'full_series':
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size = out.size(1),
                        ).squeeze(1)


                else:   
                    for g in range(len(self.grain_split_list) - 1):

                        if g == 0 :
                                encoding_x = x[ : , : , : self.grain_split_list[0]]
                                encoding_net = self.net_list[0]
                                out_g = self._eval_with_pooling(encoding_x, encoding_net, mask, encoding_window= encoding_window)
                                
                                ph_net = self.ph_list[0].to(self.device)
                                encoding_f = x[ : , : , self.grain_split_list[0] : self.grain_split_list[1]].to(self.device)
                                out_f = ph_net(encoding_f)
                            
                        else :
                            encoding_x = out_g.clone().detach().numpy()
                            ph_f = out_f.clone().detach().cpu().numpy()
                            encoding_x = np.concatenate([encoding_x, ph_f], axis= 2)
                            encoding_x = torch.from_numpy(encoding_x)
                            encoding_net = self.net_list[g]
                            out_g = self._eval_with_pooling(encoding_x, encoding_net, mask, encoding_window= encoding_window)
                            
                            ph_net = self.ph_list[g].to(self.device)
                            encoding_f = x[ : , : , self.grain_split_list[g] : self.grain_split_list[g+1]].to(self.device)
                            out_f = ph_net(encoding_f)
                    out = np.concatenate([out_g.cpu(), out_f.cpu()], axis= 2)
                    out = torch.from_numpy(out)
                    if encoding_window == 'full_series':
                        out = out.squeeze(1)
                        
                output.append(out)
                
            output = torch.cat(output, dim=0)

        for i in range(len(self.net_list)):
            self.net_list[i].train(org_training_list[i])
        
        return output.numpy()
    



    def save(self, file_name):
        ''' Save the model to a file.
        Args:
            fn (str): filename.
        '''
        param_dict = {}
        for g in range(len(self.grain_split_list) - 1):
            param_dict["encoder_" + str(g)] = self.net_list[g].state_dict()
            param_dict["projector_" + str(g)] = self.ph_list[g].state_dict()
        torch.save(param_dict, file_name)
    



    def load(self, file_name):
        ''' Load the model from a file.
        Args:
            fn (str): filename.
        '''
        param_dict = torch.load(file_name)
        for g in range(len(self.grain_split_list) - 1):
            self.net_list[g].load_state_dict(param_dict["encoder_" + str(g)])
            self.ph_list[g].load_state_dict(param_dict["projector_" + str(g)])