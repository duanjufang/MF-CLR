import torch
from torch import nn
import torch.nn.functional as F
import numpy as np




def combine_loss(z1, z2, f1, f2, alpha= 1, beta= 1e-5, verbose= False):
    loss = torch.tensor(0., device= z1.device)
    if alpha != 0 :
        loss += alpha * instance_contrastive_loss(z1, z2)
    if 1 - alpha != 0 :
        loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
    if 1 - beta != 0 :
        loss *= (1 - beta)
    if beta != 0 :
        loss += beta * feature_contrastive_loss(z1, z2, f1, f2)

    if verbose is True :
        print("z1, z2 size : ", z1.size(), z2.size())
        print("f1, f2 size : ", f1.size(), f2.size())
        print("instance : ", alpha * (1 - beta) * instance_contrastive_loss(z1, z2))
        print("temporal : ", (1 - alpha) * (1 - beta) * temporal_contrastive_loss(z1, z2))
        print("exp loss : ", beta * feature_contrastive_loss(z1, z2, f1, f2))
        print()
    return loss




def feature_contrastive_loss(z1, z2, f1, f2, t= 0.5, delta= 1, para= "quad"):

    max_featuredist = torch.tensor(1e-2, device= z1.device)
    for k in range(z1.size(0)):
        for l in range(z2.size(0)):
            max_featuredist = max(max_featuredist, torch.norm(f1[k, :, :] - f2[l, :, :]))

    if para == "exp" :
        exploss = torch.tensor(0., device= z1.device)
        for i in range(z1.size(0)):
            for j in range(z1.size(0)):
                s_ij = torch.pow(1 - torch.norm(f1[i, :, :] - f2[j, :, :]) / max_featuredist, 2.)
                D_ij = torch.norm(z1[i, :, :] - z2[j, :, :])
                exploss += torch.exp(torch.pow((1 - s_ij) * delta - D_ij, 2) / t)
        loss = t * torch.log(exploss / (int(z1.size(0)) ** 2))
        return loss

    if para == "quad" :
        loss_quad = torch.tensor(0., device= z1.device)
        for i in range(z1.size(0)):
            for j in range(z2.size(0)):
                s_ij = 1 - torch.norm(f1[i, :, :] - f2[j, :, :]) / max_featuredist
                D_ij = torch.norm(z1[i, :, :] - z2[j, :, :])
                loss_quad += torch.pow((1 - s_ij) * delta - D_ij, 2)
        loss_quad = loss_quad / (int(z1.size(0)) ** 2)
        return loss_quad




def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1 :
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim= 0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B    
    
    
    logits = torch.tril(sim, diagonal= -1)[ : , : , : -1]    # T x 2B x (2B - 1)
    logits += torch.triu(sim, diagonal= 1)[ : , : , 1 : ]    # T x 2B x (2B - 1)


    logits = -F.log_softmax(logits, dim= -1)

    
    i = torch.arange(B, device= z1.device)
    loss = (logits[ : , i, B + i - 1].mean() + logits[ : , B + i, i].mean()) / 2

    return loss




def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1 :
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim= 1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal= -1)[ :,  : , : -1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal= 1)[ : , : , 1 : ]
    logits = -F.log_softmax(logits, dim= -1)
    
    t = torch.arange(T, device= z1.device)
    loss = (logits[ : , t, T + t - 1].mean() + logits[ : , T + t, t].mean()) / 2
    return loss




if __name__ == "__main__" :

    z1 = torch.from_numpy(np.random.randint(0, 11, size= (16, 200, 10))).float()
    z2 = torch.from_numpy(np.random.randint(0, 11, size= (16, 200, 10))).float()
    f1 = torch.from_numpy(np.random.randint(0, 11, size= (16, 200, 7))).float()
    f2 = torch.from_numpy(np.random.randint(0, 11, size= (16, 200, 7))).float()
    loss = combine_loss(z1, z2, f1, f2)
    print(loss)