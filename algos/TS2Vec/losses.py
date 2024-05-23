import torch
from torch import nn
import torch.nn.functional as F



# 算 loss 的 z1 和 z2 分别是左右两个 view 的 crop 的编码器输出
# 他们的 size 都是 B X T X C
def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    d = 0

    # 每次调用 max_pool1d 都会让 size(1) 也就是时间长度 / 2
    while z1.size(1) > 1 :
        if alpha != 0 :
            loss += alpha * instance_contrastive_loss(z1, z2)   
        if d >= temporal_unit :
            if 1 - alpha != 0 :
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    
    # 当卷到最上层时
    if z1.size(1) == 1 :
        if alpha != 0 :
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d




def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1 :
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim= 0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B    
    # 这一步相乘实际上就是同一 Batch 的不同 ts 之间的 C 维（数据维）相乘，而 T 维（时长维）不变
    # 乘出来的应该是一个对称阵，每个 i,j 点都是一个 batch 中第 i 条和第 j 条在不同维度上的点积
    
    
    logits = torch.tril(sim, diagonal= -1)[ : , : , : -1]    # T x 2B x (2B - 1)
    logits += torch.triu(sim, diagonal= 1)[ : , : , 1 : ]    # T x 2B x (2B - 1)
    # 上面这个实际上就是下三角和上三角（都不要对角线）错位拼接

    logits = -F.log_softmax(logits, dim= -1)
    # 在最后一个维度上计算 log_softmax，这个已经相当于论文中的 L_inst_i_t 了
    
    i = torch.arange(B, device= z1.device)
    loss = (logits[ : , i, B + i - 1].mean() + logits[ : , B + i, i].mean()) / 2
    # logits[ : , i, B + i - 1].mean() 第二维从 0 ~ B-1，第三维从 B-1 ~ 2B-2
    # logits[ : , B + i, i].mean() 第二维从 B ~ 2B-1，第三维从 0 ~ B-1
    # 这里实际上就是相当于论文中对 i 和 t 的两个求平均
    # 第一个维度全取，是对 T 时间维度取平均
    # 第二、三维度的遍历、取得是 B 维度的。（画画就清楚了，实际上是batch1和batch2中同一个ts之间的点积）

    return loss




# 跟上面是对称的，上面操作 B 维、看个体间的；这里操作 T 维、看时刻间的
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
