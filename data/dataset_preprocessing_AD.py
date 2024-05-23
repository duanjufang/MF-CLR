import pandas as pd
import numpy as np
import os
import sys
import csv
import ast
import shutil


abs_path = os.path.dirname(os.path.realpath(__file__))




## config格式说明，分别是起始维度，终止维度，调整采样频率
# sample_config = [[sample_start, sample_end, sample_freq]]
anomaly_config = {
    "MSL" : [(1, 28, 15), (28, 55, 60)],           # min, quarter, hour
    "SMAP" : [(1, 13, 15), (13, 25, 60)],            # min, quarter, hour
    "SMD" : [(13, 26, 15), (26, 38, 60)],           # min, quarter, hour
    "SWaT" : [(25, 34, 60), (34, 51, 60*15)],       # s, min, quarter
}

SMAP_MSL_config = {
    "SMAP" : [
        "P-1", "S-1", "E-1", "E-2", "E-3", "E-4", "E-5", "E-6", "E-7", "E-8", "E-9", "E-10", "E-11", "E-12", 
        "E-13", "A-1", "D-1", "P-2", "P-3", "D-2", "D-3", "D-4", "A-2", "A-3", "A-4", "G-1", "G-2", "D-5", 
        "D-6", "D-7", "F-1", "P-4", "G-3", "T-1", "T-2", "D-8", "D-9", "F-2", "G-4", "T-3", "D-11", "D-12", 
        "B-1", "G-6", "G-7", "P-7", "R-1", "A-5", "A-6", "A-7", "D-13", "P-2", "A-8", "A-9", "F-3",
    ],
    "MSL" : [
        "M-6", "M-1", "M-2", "S-2", "P-10", "T-4", "T-5", "F-7", "M-3", "M-4", "M-5", "P-15", "C-1", "C-2",
        "T-12", "T-13", "F-4", "F-5", "D-14", "T-9", "P-14", "T-8", "P-11", "D-15", "D-16", "M-7", "F-8", 
    ],
}


abs_path = os.path.dirname(os.path.realpath(__file__))

def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                        dtype=np.float32,
                        delimiter=',')
    print(dataset, category, filename, temp.shape)
    df = pd.DataFrame(temp)
    os.makedirs("SMD"+f"/{category}", exist_ok=True)
    os.remove(os.path.join(dataset_folder, category, filename))
    filename = filename.replace(".txt", "")
    df.to_csv(os.path.join("SMD", category + f"/{filename}"+".csv"), index=None, columns=None)


def load_data(dataset):
    output_folder = f'{abs_path}/{dataset}'
    os.makedirs(dataset, exist_ok=True)

    if dataset == 'SMD':
        dataset_folder = f'{abs_path}/{dataset}'
        file_list = os.listdir(os.path.join(dataset_folder, "train"))
        for filename in file_list:
            if filename.endswith('.txt'):
                load_and_save('train', filename, filename.strip('.txt'), dataset_folder)
                load_and_save('test', filename, filename.strip('.txt'), dataset_folder)
                load_and_save('test_label', filename, filename.strip('.txt'), dataset_folder)

    elif dataset == 'SMAP' or dataset == 'MSL':
        dataset_folder = f'{abs_path}/SMAP_MSL'
        with open(os.path.join(dataset_folder, 'labeled_anomalies.csv'), 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            res = [row for row in csv_reader][1:]
        res = sorted(res, key=lambda k: k[0])
        label_folder = os.path.join(dataset_folder, 'test_label')
        os.makedirs(label_folder, exist_ok=True)
        data_info = [row for row in res if row[1] == dataset]
        labels = []
        for row in data_info:
            anomalies = ast.literal_eval(row[2])
            length = int(row[-1])
            label = np.zeros([length], dtype=np.bool)
            for anomaly in anomalies:
                label[anomaly[0]:anomaly[1] + 1] = True
            labels.extend(label)

        labels = list(map(str, labels))
        labes_str = ",\n".join(labels)
        with open(os.path.join(output_folder, dataset + "_" + 'test_label' + ".csv"), "w") as file:
            file.write(labes_str)


        def concatenate_and_save(category):
            os.makedirs(f'{abs_path}/{dataset}/{category}', exist_ok=True)
            for row in data_info:
                filename = row[0]
                temp = np.load(os.path.join(dataset_folder, category, filename + '.npy'))
                df = pd.DataFrame(temp)
                df.to_csv(os.path.join(output_folder, category + f"/{filename}"+".csv"), index=None, columns=None)



        for c in ['train', 'test']:
            concatenate_and_save(c)
            
        if os.path.exists(f'{abs_path}/SMAP/') and os.path.exists(f'{abs_path}/MSL/'):
            shutil.rmtree(f'{abs_path}/SMAP_MSL')


def resample_AD(dataset):
    assert dataset in ["SMAP", "SMD", "MSL", "SWaT"]
    
    if dataset == "SWaT" :
        train_df = pd.read_csv(f"{abs_path}" + "/SWaT/swat_train2.csv")
        train_df.drop(columns= "Normal/Attack", inplace= True)
        test_df = pd.read_csv(f"{abs_path}" + "/SWaT/swat2.csv")
        test_label = test_df["Normal/Attack"].values
        test_df.drop(columns= "Normal/Attack", inplace= True)
        print("original shape : ")
        print("train_df : ", train_df.shape)
        print("test_df : ", test_df.shape)
        print("test_label : ", test_label.shape)
        print()

        sample_config = anomaly_config[dataset]
        seg_len = sample_config[-1][-1]
        train_X = train_df.values[-(seg_len * int(train_df.shape[0] // seg_len)) : , : ]
        train_X = train_X.reshape(1, train_X.shape[0], train_X.shape[1])
        test_X = test_df.values[-(seg_len * int(test_df.shape[0] // seg_len)) : , : ]
        test_X = test_X.reshape(1, test_X.shape[0], test_X.shape[1])
        test_label = test_label[-(seg_len * int(test_df.shape[0] // seg_len)) : ]
        for each_config in sample_config :
            start, end, freq = each_config
            train_X[ : , : , start : end] = train_X[ : , range(0, train_X.shape[1], freq), start : end].repeat(freq, axis= 1)
            test_X[ : , : , start : end] = test_X[ : , range(0, test_X.shape[1], freq), start : end].repeat(freq, axis= 1)
        test_label = test_label.reshape(1, -1)
        print("resampled shape : ")
        print("train_df : ", train_X.shape)
        print("test_df : ", test_X.shape)
        print("test_label : ", test_label.shape)
        print()


    elif dataset == "SMD" :
        sample_config = anomaly_config[dataset]
        seg_len = sample_config[-1][-1]
        train_X, test_X, test_label = [], [], []
        train_list = [os.path.join(f"{abs_path}/" + dataset + "/train/", file) for file in os.listdir(f"{abs_path}/" + dataset + "/train/")]
        for train_df in train_list :
            train_x = pd.read_csv(train_df).values
            train_x = np.nan_to_num(train_x)
            train_x = train_x[-(seg_len * int(train_x.shape[0] // seg_len)) : , : ]
            train_X.append(train_x)
        test_list = [os.path.join(f"{abs_path}/" + dataset + "/test/", file) for file in os.listdir(f"{abs_path}/" + dataset + "/test/")]
        for test_df in test_list :
            test_x = pd.read_csv(test_df).values
            test_x = np.nan_to_num(test_x)
            test_x = test_x[-(seg_len * int(test_x.shape[0] // seg_len)) : , : ]
            test_X.append(test_x)
        label_list = [os.path.join(f"{abs_path}/" + dataset + "/test_label/", file) for file in os.listdir(f"{abs_path}/" + dataset + "/test_label/")]
        for label_df in label_list :
            label = pd.read_csv(label_df).values
            label = label[-(seg_len * int(label.shape[0] // seg_len)) : , : ]
            test_label.append(label)
        print("original shape : ")
        for i in range(len(train_X)):
            print(train_X[i].shape, test_X[i].shape, test_label[i].shape)
        print()
        
        train_X_resmaple, test_X_resample = [], []
        for train_x in train_X :
            train_x = train_x.reshape(1, train_x.shape[0], train_x.shape[1])
            for each_config in sample_config :
                start, end, freq = each_config
                train_x[ : , : , start : end] = train_x[ : , range(0, train_x.shape[1], freq), start : end].repeat(freq, axis= 1)
            train_x = np.squeeze(train_x)
            train_X_resmaple.append(train_x)
        for test_x in test_X :
            test_x = test_x.reshape(1, test_x.shape[0], test_x.shape[1])
            for each_config in sample_config :
                start, end, freq = each_config
                test_x[ : , : , start : end] = test_x[ : , range(0, test_x.shape[1], freq), start : end].repeat(freq, axis= 1)
            test_x = np.squeeze(test_x)
            test_X_resample.append(test_x)
        print("resampled shape : ")
        for i in range(len(train_X)):
            print(train_X_resmaple[i].shape, test_X_resample[i].shape, test_label[i].shape)
        print()
        train_X = train_X_resmaple
        test_X = test_X_resample
    

    elif dataset in ["SMAP", "MSL"] :
        label_df = pd.read_csv(f"{abs_path}/" + dataset + "/" + dataset + "_test_label.csv", header= None)
        label_df = label_df.iloc[ : , 0]
        sample_config = anomaly_config[dataset]
        seg_len = sample_config[-1][-1]
        counter = 0
        train_X, test_X, test_label = [], [], []
        dataset_config = SMAP_MSL_config[dataset]
        for each_subdata in dataset_config :
            train_df = pd.read_csv(f"{abs_path}/" + dataset + "/train/" + each_subdata + ".csv")
            train_x = train_df.values[-(seg_len * int(train_df.shape[0] // seg_len)) : , : ]
            train_x = train_x.reshape(1, train_x.shape[0], train_x.shape[1])
            test_df = pd.read_csv(f"{abs_path}/" + dataset + "/test/" + each_subdata + ".csv")
            test_x = test_df.values[-(seg_len * int(test_df.shape[0] // seg_len)) : , : ]
            if train_x.shape[1] < 500 : continue
            test_x = test_x.reshape(1, test_x.shape[0], test_x.shape[1])
            for each_config in sample_config :
                start, end, freq = each_config
                train_x[ : , : , start : end] = train_x[ : , range(0, train_x.shape[1], freq), start : end].repeat(freq, axis= 1)
                test_x[ : , : , start : end] = test_x[ : , range(0, test_x.shape[1], freq), start : end].repeat(freq, axis= 1)
            label = label_df.values[counter : counter + test_df.shape[0]]
            label = label[-test_x.shape[1] : ]
            counter += test_df.shape[0]
            train_X.append(np.squeeze(train_x))
            test_X.append(np.squeeze(test_x))
            test_label.append(label)
        print("counter = ", counter)
        print("resampled shape : ")
        for i in range(len(train_X)):
            print(train_X[i].shape, test_X[i].shape, test_label[i].shape)
        print()

    return train_X, test_X, test_label


if __name__ == "__main__" :

    datasets = ['SMD', 'SMAP', 'MSL']
    commands = sys.argv[1:]
    load = []
    if len(commands) > 0:
        for d in commands:
            if d in datasets:
                load_data(d)
    else:
        print("""
        Usage: python data_preprocess.py <datasets>
        where <datasets> should be one of ['SMD', 'SMAP', 'MSL']
        """)