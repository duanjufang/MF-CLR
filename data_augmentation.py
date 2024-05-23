import numpy as np
import pandas as pd
from pandarallel import pandarallel
pandarallel.initialize(nb_workers= 64)




class PathGeneration():
    def __init__(self, input_seq: np.array, seq_len= None):
        self.ori_x = input_seq
        if seq_len is None :
            _prob = max(-0.3, min(0.3, np.random.randn() / 10))
            self.seq_len = int(input_seq.shape[0] * (1 + _prob))
        else :
            self.seq_len = seq_len
    

    def check_dist(self, dtw_mat, idx: list):
        assert len(idx) == 2
        k = dtw_mat.shape[0] / dtw_mat.shape[1]
        x = idx[1]
        y = dtw_mat.shape[0] - 1 - idx[0]
        x_diagnose = y / k
        horizontal_dist = abs(int(x - x_diagnose))
        y_diagnose = x * k
        vertical_dist = abs(int(y - y_diagnose))
        return horizontal_dist, vertical_dist
    

    def prob_sample(self, inputDict):
        # inputDict = {value_1 : prob_1, value_2 : prob_2, ...}
        candidate_list = []
        for each_val in inputDict.keys():
            for i in range(int(inputDict[each_val] * 100)):
                candidate_list.append(each_val)
        if len(candidate_list) == 0 :
            return list(inputDict.keys())[np.random.randint(len(inputDict.keys()))]
        else :
            random_idx = np.random.randint(0, len(candidate_list))
            return candidate_list[random_idx]

    
    def cal_prob(self, dist, dist_aug, dist_param= 0.5):
        if dist_aug >= 6 : return 0
        return np.exp(-dist * dist_param * dist_aug)
    

    def find_path_opt(self):
        dtw_mat = np.zeros((self.ori_x.shape[0], self.seq_len))
        dtw_mat[-1, 0] = 1
        # print("dtw_mat.shape : ", dtw_mat.shape)

        bias_dict = {}

        i = dtw_mat.shape[0] - 1
        j = 0
        last_switch_idx = None  
        while i != 0 or j != dtw_mat.shape[1] - 1 :
            if (i, j) in bias_dict.keys() :
                this_h_dist = bias_dict[(i, j)][0]
                this_v_dist = bias_dict[(i, j)][1]
            else :
                this_h_dist, this_v_dist = self.check_dist(dtw_mat, [i, j])
                bias_dict[(i, j)] = (this_h_dist, this_v_dist)
            dist = this_h_dist ** 2 + this_v_dist ** 2
            if  i == 0 :                        
                random_switch = [0, 1]
            elif j == dtw_mat.shape[1] - 1 :    
                random_switch = [-1, 0]
            else :
                switch = [[-1, 0], [-1, 1], [0, 1]]
                if i == 1 : switch.remove([-1, 0])                      
                if j == dtw_mat.shape[1] - 2 : switch.remove([0, 1])     
                if last_switch_idx == 0 and [0, 1] in switch: switch.remove([0, 1])         
                if last_switch_idx == 2 and [-1, 0] in switch: switch.remove([-1, 0])       
                dist_update = []
                for each_aug in switch :
                    i_idx = i + each_aug[0]
                    j_idx = j + each_aug[1]
                    if (i_idx, j_idx) in bias_dict.keys() :
                        next_h_dist = bias_dict[(i_idx, j_idx)][0]
                        next_v_dist = bias_dict[(i_idx, j_idx)][1]
                    else :
                        next_h_dist, next_v_dist = self.check_dist(dtw_mat, [i_idx, j_idx])
                        bias_dict[(i, j)] = (next_h_dist, next_v_dist)
                    next_dist = next_h_dist ** 2 + next_v_dist ** 2
                    dist_aug = next_dist - dist
                    dist_update.append(dist_aug)
                switch_idx_sample_dict = {}
                for each_dist_aug_idx in range(len(dist_update)) :
                    switch_idx_sample_dict[each_dist_aug_idx] = self.cal_prob(dist, dist_update[each_dist_aug_idx])
                random_switch_idx = self.prob_sample(switch_idx_sample_dict)
                random_switch = switch[random_switch_idx]
                last_switch_idx = random_switch_idx
            i += random_switch[0]
            j += random_switch[1]
            dtw_mat[i, j] = 1
        return dtw_mat
    
    
    def find_path(self):
        dtw_mat = np.zeros((self.ori_x.shape[0], self.seq_len))
        dtw_mat[-1, 0] = 1
        # print("dtw_mat.shape : ", dtw_mat.shape)

        
        h_bias_mat, v_bias_mat = dtw_mat.copy(), dtw_mat.copy()
        for i in range(dtw_mat.shape[0]):
            for j in range(dtw_mat.shape[1]):
                h_bias_mat[i, j], v_bias_mat[i, j] = self.check_dist(dtw_mat, [i, j])

        
        i = dtw_mat.shape[0] - 1
        j = 0
        last_switch_idx = None  
        while i != 0 or j != dtw_mat.shape[1] - 1 :
            this_h_dist = h_bias_mat[i, j]
            this_v_dist = v_bias_mat[i, j]
            dist = this_h_dist ** 2 + this_v_dist ** 2
            if  i == 0 :                        
                random_switch = [0, 1]
            elif j == dtw_mat.shape[1] - 1 :    
                random_switch = [-1, 0]
            else :
                switch = [[-1, 0], [-1, 1], [0, 1]]
                if i == 1 : switch.remove([-1, 0])                      
                if j == dtw_mat.shape[1] - 2 : switch.remove([0, 1])     
                if last_switch_idx == 0 and [0, 1] in switch: switch.remove([0, 1])         
                if last_switch_idx == 2 and [-1, 0] in switch: switch.remove([-1, 0])       
                dist_update = []
                for each_aug in switch :
                    i_idx = i + each_aug[0]
                    j_idx = j + each_aug[1]
                    next_h_dist = h_bias_mat[i_idx, j_idx]
                    next_v_dist = v_bias_mat[i_idx, j_idx]
                    next_dist = next_h_dist ** 2 + next_v_dist ** 2
                    dist_aug = next_dist - dist
                    dist_update.append(dist_aug)
                switch_idx_sample_dict = {}
                for each_dist_aug_idx in range(len(dist_update)) :
                    switch_idx_sample_dict[each_dist_aug_idx] = self.cal_prob(dist, dist_update[each_dist_aug_idx])
                random_switch_idx = self.prob_sample(switch_idx_sample_dict)
                random_switch = switch[random_switch_idx]
                last_switch_idx = random_switch_idx
            i += random_switch[0]
            j += random_switch[1]
            dtw_mat[i, j] = 1
        return dtw_mat




class AmpGeneration(PathGeneration):
    def __init__(self, input_seq: np.array, seq_len=None, total_dist= None):
        super().__init__(input_seq, seq_len)
        if total_dist is None :
            _prob = max(-0.4, min(0.4, np.random.randn() / 7))
            self.total_dist = int(input_seq.shape[0] * (1 + _prob))
        else :
            self.total_dist = total_dist  
    

    def dist_distribution(self, total= None, number= None):
        if total is None : total = self.total_dist
        if number is None : number = self.seq_len
        # acc_list = [np.random.randint(0, total) for i in range(number - 1)]
        acc_list = [np.random.uniform(low= 0, high= total) for i in range(number - 1)]
        acc_list.append(0)
        acc_list.append(total)
        acc_list.sort()
        interval_list = [acc_list[i + 1] - acc_list[i] for i in range(number)]
        return interval_list
    

    def find_next_idx(self, dtw_mat, i, j):
        for each_move in [[-1, 0], [-1, 1], [0, 1]]:
            i_next = i + each_move[0]
            j_next = j + each_move[1]
            if dtw_mat[i_next, j_next] == 1 :
                return i_next, j_next, each_move
    

    def seq_generation(self, path_mat):
        rows = path_mat.shape[0]
        cols = path_mat.shape[1]
        ori_seq = self.ori_x
        gen_seq = [None for i in range(cols)]
        i = rows - 1
        j = 0
        counter = 0
        interval_list = self.dist_distribution(number= int(np.sum(path_mat.reshape(-1, ))))
        # print(interval_list)

        while True :
            ori_seq_idx = rows - i - 1
            gen_seq_idx = j
            # print(ori_seq_idx, gen_seq_idx, counter, i != 0, j != cols - 1, interval_list[counter])
            
            if gen_seq[gen_seq_idx] == None :
                gen_seq[gen_seq_idx] = ori_seq[ori_seq_idx] + self.prob_sample({1 : 0.5, -1 : 0.5}) * interval_list[counter]
                sub_counter = 1
            else :
                update_gen_val = ori_seq[ori_seq_idx] + interval_list[counter]
                # print(gen_seq_idx, update_gen_val)
                gen_seq[gen_seq_idx] = sub_counter * gen_seq[gen_seq_idx] / (sub_counter + 1) + update_gen_val / (sub_counter + 1)
                sub_counter += 1

            if i == 0 and j == cols - 1 : break

            i_next, j_next, _ = self.find_next_idx(path_mat, i, j)
            i = i_next
            j = j_next
            counter += 1
            # print(gen_seq)

        total_pair_num = np.sum(path_mat.reshape(-1, ))
        # print("total_pair_num : ", total_pair_num, counter)
        return gen_seq




def data_generation(inputArray):    # inputArray: B * T * D
    assert np.ndim(inputArray) == 3
    DA_array = []
    for each_b in range(inputArray.shape[0]):
        batch_genarray = []
        for each_d in range(inputArray.shape[2]):
            sub_array = inputArray[each_b, : , each_d]
            sub_array = sub_array.reshape(-1, )
            DA = AmpGeneration(input_seq= sub_array, seq_len= sub_array.shape[0], total_dist= 0.05*np.mean(sub_array))
            path_mat = DA.find_path()
            print(each_b, each_d, inputArray.shape[0], inputArray.shape[2])
            gen_array = DA.seq_generation(path_mat)
            batch_genarray.append(gen_array)
        DA_array.append(batch_genarray)
    DA_array = np.array(DA_array)
    DA_array = DA_array.transpose((0, 2, 1))
    return DA_array




def gen_lambda(inputArray, b_idx, d_idx):
    sub_array = inputArray[b_idx, : , d_idx]
    sub_array = sub_array.reshape(-1, )
    DA = AmpGeneration(input_seq= sub_array, seq_len= sub_array.shape[0], total_dist= 0.05*np.mean(sub_array))
    path_mat = DA.find_path_opt()
    gen_array = DA.seq_generation(path_mat)
    return gen_array




def data_generation_multi(inputArray):    # inputArray: B * T * D
    
    assert np.ndim(inputArray) == 3
        
    temp_df = pd.DataFrame()
    b_list, d_list = [], []
    for each_b in range(inputArray.shape[0]):
        for each_d in range(inputArray.shape[2]):
            b_list.append(each_b)
            d_list.append(each_d)
    temp_df["b"] = b_list
    temp_df["d"] = d_list
    temp_df["arr"] = temp_df.parallel_apply(lambda x : gen_lambda(inputArray, x["b"], x["d"]), axis= 1)
    DA_array = [[None for d in range(inputArray.shape[2])] for b in range(inputArray.shape[0])]
    counter = 0
    for each_b in range(inputArray.shape[0]):
        for each_d in range(inputArray.shape[2]):
            DA_array[each_b][each_d] = temp_df["arr"].values[counter]
            counter += 1
    DA_array = np.array(DA_array)
    DA_array = DA_array.transpose((0, 2, 1))
    return DA_array




if __name__ == "__main__" :

    input_seq = np.random.randint(11, 44, size= (53, 2222, 10)) / 100
    # input_seq = input_seq.reshape(3, -1, 11)
    print("input_seq : ", input_seq.shape)
    
    # pg_class = AmpGeneration(input_seq, seq_len= input_seq.shape[0])
    # dtw_mat = pg_class.find_path()
    # print(dtw_mat)

    # gen_seq = pg_class.seq_generation(dtw_mat)
    # print(len(input_seq.tolist()), input_seq)
    # print(len(gen_seq), gen_seq)

    from time import time
    print("start data augmentations")
    time_start = time()
    gen_seq = data_generation(input_seq)
    print(gen_seq.shape)
    print(f"\n*** Time cost: {round((time() - time_start)/60, 3)}min")


    # import matplotlib.pyplot as plt
    # plt.figure(figsize= [11, 4])
    # x = np.arange(gen_seq.shape[1])
    # plt.plot(x, input_seq.reshape(-1, ) * 100, label= "original")
    # plt.plot(x, gen_seq.reshape(-1, ) * 100, label= "augmented")
    # plt.ylim(0, 44)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    # plt.savefig("temp_DA.jpg")