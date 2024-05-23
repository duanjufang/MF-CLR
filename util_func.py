import numpy as np
import torch




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




def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs





def take_per_row(A, indx, num_elem, split_point):
    all_indx = indx[ : , None] + np.arange(num_elem)
    data_per_row = A[torch.arange(all_indx.shape[0])[ : , None], all_indx]
    return data_per_row[ : , : , : split_point], data_per_row[ : , : , split_point : ]
    





def take_per_row_multigrain(A, indx, num_elem, split_list):
    all_indx = indx[ : , None] + np.arange(num_elem)
    data_per_row = A[torch.arange(all_indx.shape[0])[ : , None], all_indx]
    grain_data_list = []
    for i in range(len(split_list)) :
        if i == 0 :
            grain_data_list.append(data_per_row[ : , : , 0 : split_list[i]])
        else :
            grain_data_list.append(data_per_row[ : , : , split_list[i - 1] : split_list[i]])
    return grain_data_list




def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis= -1), axis= 1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis= -1), axis= 1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[ : x.shape[0], : x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[ : , np.newaxis]
    return x[rows, column_indices]





def torch_pad_nan(arr, left= 0, right= 0, dim= 0):
    if left > 0:
        padshape = list(arr.shape)
        padshape[dim] = left
        arr = torch.cat((torch.full(padshape, np.nan), arr), dim= dim)
    if right > 0:
        padshape = list(arr.shape)
        padshape[dim] = right
        arr = torch.cat((arr, torch.full(padshape, np.nan)), dim= dim)
    return arr




def generate_pred_samples(features, data, pred_len, drop=0):

    n = data.shape[1]                       
    features = features[ : , : -pred_len]  
    labels = np.stack([data[ : , i : 1 + n + i - pred_len] for i in range(pred_len)], axis= 2)[ : , 1 : ]   # B * T_train * T_pred * D
    features = features[ : , drop : ]
    labels = labels[ : , drop : ]   # B * (T_train - drop) * T_pred * D
    print(features.shape)
    print(labels.shape)
    return features.reshape(-1, features.shape[-1]), labels.reshape(-1, labels.shape[2] * labels.shape[3])  




def cal_metrics(pred, target):
    return {
        'MSE': ((pred - target) ** 2).mean(),
        'MAE': np.abs(pred - target).mean()
    }




if __name__ == "__main__" :

    # a_tensor = torch.from_numpy(np.random.randint(0, 10, size= (1, 7, 5)))
    # print(a_tensor)
    # print()
    # crop_offset = np.random.randint(low= 0, high= 2, size= a_tensor.size(0))
    # croped = take_per_row(a_tensor, crop_offset + 1, 4, 2)
    # print(crop_offset + 1)
    # print(croped)

    arr = torch.from_numpy(np.random.randint(0, 10, size= (8, 20, 4)))
    print(arr)
    print()
    arr_after = torch_pad_nan(arr, 2, 2, 1)
    print(arr_after)
    print()
    print(arr.size())
    print(arr_after.size())