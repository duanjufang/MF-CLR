import os
import numpy as np
import pickle
import torch
import random
from datetime import datetime




def pkl_save(name, var):
    with open(name, 'wb') as f:
        pickle.dump(var, f)




def pkl_load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
    



def torch_pad_nan(arr, left=0, right=0, dim=0):
    if left > 0:
        padshape = list(arr.shape)
        padshape[dim] = left
        arr = torch.cat((torch.full(padshape, np.nan), arr), dim=dim)
    if right > 0:
        padshape = list(arr.shape)
        padshape[dim] = right
        arr = torch.cat((arr, torch.full(padshape, np.nan)), dim=dim)
    return arr
    



def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size//2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)




# 将 x 分割为 sections 这么多段
# 最后不够长的调用 pad_nan_to_target 补为 np.nan
def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs




# 这个函数的目的实际上就是 crop
# 被截取对象是 A，对应主函数的 x，是三维的，第一维是每个 batch 的条数，第二维是长度，第三维是数据维度，B * T * C
# 开始的点是 indx
# 截取的长度是 num_elem
# 截取后第一维和第三维不变，只在第二维（时间步）上做 crop
def take_per_row(A, indx, num_elem):
    all_indx = indx[ : , None] + np.arange(num_elem)
    # indx[ :, None] 相当于多加了一个维度，把一维的 (xxx, ) 转化为二维 (xxx, 1)
    # 再 + np.arange(num_elem) 维数转为 (xxx, num_elem)，每一行的那一个数都广播着加了个 np.arange()

    return A[torch.arange(all_indx.shape[0])[ : , None], all_indx]
    # 把 torch.arange(all_indx.shape[0])[ : , None] 这个生成的二维顺序索引, 把它变成二维的索引，取出来的也是二维的，相当于 [] 起来的一维切片
    # 和刚才生成的 all_indx
    # 都挤到一个 list 中
    # 然后前面的顺序索引取 A 的行，后面的 all_indx 取 A 的列




def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis= -1), axis= 1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis= -1), axis= 1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[ : x.shape[0], : x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[ : , np.newaxis]
    return x[rows, column_indices]




def data_dropout(arr, p):
    B, T = arr.shape[0], arr.shape[1]
    mask = np.full(B*T, False, dtype=np.bool)
    ele_sel = np.random.choice(B * T, size= int(B * T * p), replace= False
    )
    mask[ele_sel] = True
    res = arr.copy()
    res[mask.reshape(B, T)] = np.nan
    return res




def name_with_datetime(prefix='default'):
    now = datetime.now()
    return prefix + '_' + now.strftime("%Y%m%d_%H%M%S")




def init_dl_program(
    device_name,
    seed=None,
    use_cudnn=True,
    deterministic=False,
    benchmark=False,
    use_tf32=False,
    max_threads=None
):
    import torch
    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)
        
    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)
        
    if isinstance(device_name, (str, int)):
        device_name = [device_name]
    
    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        
    return devices if len(devices) > 1 else devices[0]

