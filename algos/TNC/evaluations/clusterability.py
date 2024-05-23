"""
Measure clusterability performance on multiple baselines and for different datasets
"""

import torch
import os
from tnc.models import WFEncoder, RnnEncoder
import pickle
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans


device = 'cuda' if torch.cuda.is_available() else 'cpu'

### Clusterability on Waveform
encoder = WFEncoder(encoding_size=64)
window_size = 2500
datapath = './data/waveform_data/processed'
with open(os.path.join(datapath, 'x_test.pkl'), 'rb') as f:
    x_test = pickle.load(f)
with open(os.path.join(datapath, 'state_test.pkl'), 'rb') as f:
    y_test = pickle.load(f)

T = x_test.shape[-1]
x_chopped_test = np.split(x_test[:, :, :window_size * (T // window_size)], (T // window_size), -1)
y_chopped_test = np.concatenate(np.split(y_test[:, :window_size * (T // window_size)], (T // window_size), -1),
                                   0).astype(int)
x_chopped_test = torch.Tensor(np.concatenate(x_chopped_test, 0))
y_chopped_test = torch.Tensor(np.array([np.bincount(yy).argmax() for yy in y_chopped_test]))
testset = torch.utils.data.TensorDataset(x_chopped_test, y_chopped_test)
loader = torch.utils.data.DataLoader(testset, batch_size=100)

n_test = len(x_test)
inds = np.random.randint(0, x_test.shape[-1] - window_size, n_test * 200)
windows = np.array([x_test[int(i % n_test), :, ind:ind + window_size] for i, ind in enumerate(inds)])
windows = torch.Tensor(windows).to(device)
y_window = np.array([y_test[i % n_test, ind:ind + window_size] for i, ind in enumerate(inds)]).astype(int)
windows_state = np.array([np.bincount(yy).argmax() for yy in y_window])

print('\nWAVEFORM DATASET')
for i, path in enumerate(['waveform','waveform_cpc','waveform_trip']):
    print('Score for ', path)
    s_score = []
    db_score = []
    for cv in range(3):
        checkpoint = torch.load('./ckpt/%s/checkpoint_%d.pth.tar'%(path, cv))
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        encoder = encoder.to(device)
        encoder.eval()
        encodings = []
        for windows, _ in loader:
            windows = windows.to(device)
            encoding = encoder(windows).detach().cpu().numpy()
            encodings.append(encoding)
        encodings = np.concatenate(encodings, 0)
        kmeans = KMeans(n_clusters=4, random_state=1).fit(encodings)
        cluster_labels = kmeans.labels_
        s_score.append(silhouette_score(encodings, cluster_labels))
        db_score.append(davies_bouldin_score(encodings, cluster_labels))
        del encodings
    print('Silhouette score: ', np.mean(s_score),'+-', np.std(s_score))
    print('Davies Bouldin score: ', np.mean(db_score),'+-', np.std(db_score))


### Clusterability on Simulation
encoder = RnnEncoder(hidden_size=100, in_channel=3, encoding_size=10, device=device)
window_size = 50
datapath = './data/simulated_data/'
with open(os.path.join(datapath, 'x_test.pkl'), 'rb') as f:
    x_test = pickle.load(f)
with open(os.path.join(datapath, 'state_test.pkl'), 'rb') as f:
    y_test = pickle.load(f)

T = x_test.shape[-1]
x_chopped_test = np.split(x_test[:, :, :window_size * (T // window_size)], (T // window_size), -1)
y_chopped_test = np.concatenate(np.split(y_test[:, :window_size * (T // window_size)], (T // window_size), -1),
                                   0).astype(int)
x_chopped_test = torch.Tensor(np.concatenate(x_chopped_test, 0))
y_chopped_test = torch.Tensor(np.array([np.bincount(yy).argmax() for yy in y_chopped_test]))
testset = torch.utils.data.TensorDataset(x_chopped_test, y_chopped_test)
loader = torch.utils.data.DataLoader(testset, batch_size=100)

n_test = len(x_test)
inds = np.random.randint(0, x_test.shape[-1] - window_size, n_test * 100)
windows = np.array([x_test[int(i % n_test), :, ind:ind + window_size] for i, ind in enumerate(inds)])
windows = torch.Tensor(windows).to(device)
y_window = np.array([y_test[i % n_test, ind:ind + window_size] for i, ind in enumerate(inds)]).astype(int)
windows_state = np.array([np.bincount(yy).argmax() for yy in y_window])

print('\nSIMULATION DATASET')
for i, path in enumerate(['simulation','simulation_cpc','simulation_trip']):
    print('Score for ', path)
    s_score = []
    db_score = []
    for cv in range(4):
        if not os.path.exists('./ckpt/%s/checkpoint_%d.pth.tar'%(path, cv)):
            continue
        checkpoint = torch.load('./ckpt/%s/checkpoint_%d.pth.tar'%(path, cv))
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        encoder = encoder.to(device)
        encoder.eval()
        encodings = []
        for windows, _ in loader:
            windows = windows.to(device)
            encoding = encoder(windows).detach().cpu().numpy()
            encodings.append(encoding)
        encodings = np.concatenate(encodings, 0)

        kmeans = KMeans(n_clusters=4, random_state=1).fit(encodings)
        cluster_labels = kmeans.labels_
        s_score.append(silhouette_score(encodings, cluster_labels))
        db_score.append(davies_bouldin_score(encodings, cluster_labels))
        del encodings
    print('Silhouette score: ', np.mean(s_score),'+-', np.std(s_score))
    print('Davies Bouldin score: ', np.mean(db_score),'+-', np.std(db_score))


### Clusterability on HAR data
encoder = RnnEncoder(hidden_size=100, in_channel=561, encoding_size=10, device=device)
window_size = 5
datapath = './data/HAR_data/'
with open(os.path.join(datapath, 'x_test.pkl'), 'rb') as f:
    x_test = pickle.load(f)
with open(os.path.join(datapath, 'state_test.pkl'), 'rb') as f:
    y_test = pickle.load(f)

T = x_test.shape[-1]
x_chopped_test = np.split(x_test[:, :, :window_size * (T // window_size)], (T // window_size), -1)
y_chopped_test = np.concatenate(np.split(y_test[:, :window_size * (T // window_size)], (T // window_size), -1),
                                   0).astype(int)
x_chopped_test = torch.Tensor(np.concatenate(x_chopped_test, 0))
y_chopped_test = torch.Tensor(np.array([np.bincount(yy).argmax() for yy in y_chopped_test]))
testset = torch.utils.data.TensorDataset(x_chopped_test, y_chopped_test)
loader = torch.utils.data.DataLoader(testset, batch_size=100)

n_test = len(x_test)
inds = np.random.randint(0, x_test.shape[-1] - window_size, n_test * 100)
windows = np.array([x_test[int(i % n_test), :, ind:ind + window_size] for i, ind in enumerate(inds)])
windows = torch.Tensor(windows).to(device)
y_window = np.array([y_test[i % n_test, ind:ind + window_size] for i, ind in enumerate(inds)]).astype(int)
windows_state = np.array([np.bincount(yy).argmax() for yy in y_window])

print('\nHAR DATASET')
for i, path in enumerate(['har','har_cpc','har_trip']):
    print('Score for ', path)
    s_score = []
    db_score = []
    for cv in range(4):
        if not os.path.exists('./ckpt/%s/checkpoint_%d.pth.tar'%(path, cv)):
            continue
        checkpoint = torch.load('./ckpt/%s/checkpoint_%d.pth.tar'%(path, cv))
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        encoder = encoder.to(device)
        encoder.eval()
        encodings = []
        for windows, _ in loader:
            windows = windows.to(device)
            encoding = encoder(windows).detach().cpu().numpy()
            encodings.append(encoding)
        encodings = np.concatenate(encodings, 0)

        kmeans = KMeans(n_clusters=6, random_state=1).fit(encodings)
        cluster_labels = kmeans.labels_
        s_score.append(silhouette_score(encodings, cluster_labels))
        db_score.append(davies_bouldin_score(encodings, cluster_labels))
        del encodings
    print('Silhouette score: ', np.mean(s_score),'+-', np.std(s_score))
    print('Davies Bouldin score: ', np.mean(db_score),'+-', np.std(db_score))