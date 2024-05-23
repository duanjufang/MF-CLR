import os
import pickle
import numpy as np
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler


abs_path = os.path.dirname(os.path.realpath(__file__))




def load_UEA(dataset):
    
    train_data = loadarff(f'{abs_path}/UEA/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'{abs_path}/UEA/{dataset}/{dataset}_TEST.arff')[0]
    

    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([d.tolist() for d in t_data ])
            
            # t_data = np.array([t[~np.isnan(t)] for t in t_data])
            # print(t_data.shape)

            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.nan_to_num(np.array(res_data).swapaxes(1, 2)), np.array(res_labels)
    

    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)
    
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    
    labels = np.unique(train_y)
    transform = { k : i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)

    return train_X, train_y, test_X, test_y


    

## config格式说明，分别是起始维度，终止维度，调整采样频率
# sample_config = [[sample_start, sample_end, sample_freq]]
UEA_config = {
    "ArticularyWordRecognition" : [(3, 6, 2), (6, 9, 3)],
    "AtrialFibrillation" : [(1, 2, 2)],
    "BasicMotions" : [(3, 6, 2)],
    "CharacterTrajectories" : [(2, 3, 2)],
    "Cricket" : [(3, 6, 3)],
    "DuckDuckGeese" : [(500, 1000, 2), (1000, 1345, 3)],
    "EigenWorms" : [(3, 6, 4)],
    "Epilepsy" : [(2, 3, 2)],
    "ERing" : [(2, 4, 5)],
    "EthanolConcentration" : [(2, 3, 17)],
    "FaceDetection" : [(72, 144, 2)],
    "FingerMovements" : [(19, 28, 2)],
    "HandMovementDirection" : [(5, 8, 2),(8, 10, 4)],
    "Handwriting" : [(2, 3, 2)],
    "Heartbeat" : [(21, 41, 3), (41, 61, 5)],
    "InsectWingbeat" : [(100, 200, 2)],
    "JapaneseVowels" : [(6, 12, 2)],
    "Libras" : [(1, 2, 3)],
    "LSST" : [(2, 4, 2), (4, 6, 3)],
    "MotorImagery" : [(16, 32, 2), (32, 48, 3), (48, 64, 10)],
    "NATOPS" : [(12, 24, 3)],
    "PEMS-SF" : [(321, 642, 2), (642, 963, 3)],
    "PenDigits" : [(1, 2, 2)],
    "PhonemeSpectra" : [(6, 11, 7)],
    "RacketSports" : [(3, 6, 2)],
    "SelfRegulationSCP1" : [(2, 6, 4)],
    "SelfRegulationSCP2" : [(2, 7, 4)],
    "SpokenArabicDigits" : [(6, 13, 3)],
    "StandWalkJump" : [(2, 4, 5)],
    "UWaveGestureLibrary" : [(1, 2, 3), (2, 3, 5)],
}


def UEA_resample(dataset, verbose= False):

    train_X, train_y, test_X, test_y = load_UEA(dataset)
    if verbose is True :
        print("-------------------------")
        print("train_X : ")
        print(train_X.shape)
        print(train_X[ : 5, : 5, : 5])
        print()
        print("test_X : ")
        print(test_X.shape)
        print(test_X[ : 5, : 5, : 5])


    # resample
    sample_config = UEA_config[dataset]
    for each_config in sample_config:
        if dataset == "JapaneseVowels" :
            train_X = train_X[ : , : -1, :]
            test_X = test_X[ : , : -1, :]
        start, end, freq = each_config
        train_X[ : , : , start : end] = train_X[ : , range(0, train_X.shape[1], freq), start : end].repeat(freq, axis= 1)
        test_X[ : , : , start : end] = test_X[ : , range(0, test_X.shape[1], freq), start : end].repeat(freq, axis= 1)


    if verbose is True : 
        print("+++++++++++++++++++++++++")
        print("train_X : ")
        print(train_X.shape)
        print(train_X[ : 5, : 5, : 5])
        print()
        print("test_X : ")
        print(test_X.shape)
        print(test_X[ : 5, : 5, : 5])

    return train_X, train_y, test_X, test_y




if __name__ == "__main__" :

    dataset_list = [
        "ArticularyWordRecognition", 
        "AtrialFibrillation",
        "BasicMotions",
        "CharacterTrajectories",
        "Cricket",
        "DuckDuckGeese",
        "EigenWorms",
        "Epilepsy",
        "ERing",
        "EthanolConcentration",
        "FaceDetection",
        "FingerMovements",
        "HandMovementDirection",
        "Handwriting",
        "Heartbeat",
        "InsectWingbeat",
        "JapaneseVowels",
        "Libras",
        "LSST",
        "MotorImagery",
        "NATOPS",
        "PEMS-SF",
        "PenDigits",
        "PhonemeSpectra",
        "RacketSports",
        "SelfRegulationSCP1",
        "SelfRegulationSCP2",
        "SpokenArabicDigits",
        "StandWalkJump",
        "UWaveGestureLibrary",
    ]
    for dataset in dataset_list :
        print()
        print()
        print()
        print()
        print("dataset : ", dataset)
        train_X, train_y, test_X, test_y = UEA_resample(dataset)
        print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

        # with open(f'/data/duanjf2/UEA_resampled/{dataset}_trainX.csv', mode= "wb") as fw :
        #     pickle.dump(train_X, fw)
        #     fw.close()
        # with open(f'/data/duanjf2/UEA_resampled/{dataset}_trainY.csv', mode= "wb") as fw :
        #     pickle.dump(train_y, fw)
        #     fw.close()
        # with open(f'/data/duanjf2/UEA_resampled/{dataset}_testX.csv', mode= "wb") as fw :
        #     pickle.dump(test_X, fw)
        #     fw.close()
        # with open(f'/data/duanjf2/UEA_resampled/{dataset}_testY.csv', mode= "wb") as fw :
        #     pickle.dump(test_y, fw)
        #     fw.close()
