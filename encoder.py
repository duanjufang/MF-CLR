import torch
from torch import nn
import torch.nn.functional as F
import numpy as np




class SamePadConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation= 1, groups= 1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding= padding,
            dilation= dilation,
            groups= groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out
    



class ConvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final= False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual




class DilatedConvEncoder(nn.Module):
    
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        

    def forward(self, x):
        return self.net(x)




class BackboneEncoder(nn.Module):

    def __init__(self, input_dims, output_dims, hidden_dims, depth, maskFlag= True):
        super().__init__()
        self.input_dim = input_dims
        self.output_dim = output_dims
        self.hidden_dim = hidden_dims
        self.input_embed = nn.Linear(input_dims, hidden_dims)
        self.encoding = DilatedConvEncoder(hidden_dims, 
                                           [hidden_dims] * depth + [output_dims], 
                                           kernel_size= 3)
        self.dropout = nn.Dropout(p= 0.1)

    
    def Masking(self, dim_B, dim_T, prob= 0.5):
        masked = np.random.binomial(1, prob, size=(dim_B, dim_T))
        return torch.from_numpy(masked)
    

    def forward(self, x, mask= False):   # B * T * input_dim
        
        nan_mask = ~x.isnan().any(axis= -1) 
        x[~nan_mask] = 0
        x = self.input_embed(x)  # B * T * hidden_dim

        if mask is True :
            dim_B = x.size(0)
            dim_T = x.size(1)
            masked = self.Masking(dim_B, dim_T)
            masked &= nan_mask
            x[~masked] = 0
        
        x = x.transpose(1, 2)   # B * hidden_dim * T
        x = self.encoding(x)    # B * output_dim * T
        x = self.dropout(x)
        x = x.transpose(1, 2)   # B * T * output_dim

        return x
    



class Projector(nn.Module):

    def __init__(self, input_dims, h1, h2):
        super().__init__()
        self.ph = nn.Sequential(
            nn.Linear(input_dims, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
        )


    def forward(self, x):
        return self.ph(x)




if __name__ == "__main__" :


    # coarse_grain = torch.from_numpy(np.random.random(size= [8, 16, 4])).float()
    # projection_head = Projector(input_dims= coarse_grain.size(2))
    # exp_fea = projection_head(coarse_grain)
    # print(exp_fea.size())

    model = BackboneEncoder(
        input_dims= 7,
        output_dims= 64,
        hidden_dims= 64,
        depth= 4,
    )
    total = sum([param.nelement() for param in model.parameters()])
    print(total)