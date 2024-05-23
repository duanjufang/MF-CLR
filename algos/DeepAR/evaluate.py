import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

import utils
import model.net as net
from dataloader import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

abs_path = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger('DeepAR.Eval')

parser = argparse.ArgumentParser()
parser.add_argument('--is-training', default='True', help='Whether to retrain the model')
parser.add_argument('--origin-datasets', default='PUBLIC_DATASETS', help='Parent dir of the origin datasets')
parser.add_argument('--origin-dataset', default='ETTm1_proceesed_OT=quarterly.csv', help='Name of the origin dataset')
parser.add_argument('--data-folder', default='data', help='Parent dir of the dataset')
parser.add_argument('--dataset-processed', default='elect', help='Name of the dataset')
parser.add_argument('--model-name', default='DeepAR', help='Directory containing params.json')
parser.add_argument('--pred-length', default=48, help='Prediction length for each tasks')
parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--sampling', action='store_true', help='Whether to sample during evaluation')
parser.add_argument('--save-best', action='store_true', help='Whether to save best ND to param_search.txt')
parser.add_argument('--restore-file', default='best',
                    help='Optional, name of the file in --model_dir containing weights to reload before \
                    training')  # 'best' or 'epoch_#'


def evaluate(model, loss_fn, test_loader, params, plot_num, sample=True):
    '''Evaluate the model on the test set.
    Args:
        model: (torch.nn.Module) the Deep AR model
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        test_loader: load test data and labels
        params: (Params) hyperparameters
        plot_num: (-1): evaluation from evaluate.py; else (epoch): evaluation on epoch
        sample: (boolean) do ancestral sampling or directly use output mu from last time step
    '''
    model.eval()
    with torch.no_grad():
    #   plot_batch = np.random.randint(len(test_loader)-1)

      summary_metric = {}
      raw_metrics = utils.init_metrics(sample=sample)

      # Test_loader: 
      # test_batch ([batch_size, train_window, 1+cov_dim]): z_{0:T-1} + x_{1:T}, note that z_0 = 0;
      # id_batch ([batch_size]): one integer denoting the time series id;
      # v ([batch_size, 2]): scaling factor for each window;
      # labels ([batch_size, train_window]): z_{1:T}.
      for i, (test_batch, id_batch, v, labels) in enumerate(tqdm(test_loader)):
          test_batch = test_batch.permute(1, 0, 2).to(torch.float32).to(params.device)
          id_batch = id_batch.unsqueeze(0).to(params.device)
          v_batch = v.to(torch.float32).to(params.device)
          labels = labels.to(torch.float32).to(params.device)
          batch_size = test_batch.shape[1]
          input_mu = torch.zeros(batch_size, params.test_predict_start, device=params.device) # scaled
          input_sigma = torch.zeros(batch_size, params.test_predict_start, device=params.device) # scaled
          hidden = model.init_hidden(batch_size)
          cell = model.init_cell(batch_size)

          for t in range(params.test_predict_start):
            # if z_t is missing, replace it by output mu from the last time step
            zero_index = (test_batch[t,:,0] == 0)
            if t > 0 and torch.sum(zero_index) > 0:
                test_batch[t,zero_index,0] = mu[zero_index]

            mu, sigma, hidden, cell = model(test_batch[t].unsqueeze(0), id_batch, hidden, cell)
            input_mu[:,t] = v_batch[:, 0] * mu + v_batch[:, 1]
            input_sigma[:,t] = v_batch[:, 0] * sigma

          if sample:
            samples, sample_mu, sample_sigma = model.test(test_batch, v_batch, id_batch, hidden, cell, sampling=True)
            raw_metrics = utils.update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, labels, params.test_predict_start, samples, relative = params.relative_metrics)
          else:
            sample_mu, sample_sigma = model.test(test_batch, v_batch, id_batch, hidden, cell)
            raw_metrics = utils.update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, labels, params.test_predict_start, relative = params.relative_metrics)
            torch.save(labels, f"{abs_path}/experiments/DeepAR/labels.pth")
            torch.save(sample_mu, f"{abs_path}/experiments/DeepAR/sample_mu.pth")
            print(f"Actual value: {labels}")
            print(f"Predict value: {sample_mu}")

      summary_metric = utils.final_metrics(raw_metrics, sampling=sample)
      metrics_string = '; '.join('{}: {:05.3f}'.format(k, v) for k, v in summary_metric.items())
      logger.info('- Full test metrics: ' + metrics_string)
    return summary_metric