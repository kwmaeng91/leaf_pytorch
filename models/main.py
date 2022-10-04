"""Script to run the baselines."""
import argparse
import importlib
import numpy as np
import os
import sys
import random
import torch
import torch.optim as optim
import torch.nn as nn
import math

import metrics.writer as metrics_writer

from baseline_constants import MAIN_PARAMS, MODEL_PARAMS
from client import Client
from server import Server

from utils.args import parse_args
from utils.model_utils import read_data

STAT_METRICS_PATH = 'metrics/stat_metrics.csv'
SYS_METRICS_PATH = 'metrics/sys_metrics.csv'

def main(args):
    print(args)

    # Set the random seed if provided (affects client sampling, and batching)
    random.seed(1 + args.seed)
    np.random.seed(12 + args.seed)
    #tf.set_random_seed(123 + args.seed)
    torch.manual_seed(123 + args.seed)

    if args.use_gpu and not torch.cuda.is_available():
        raise AssertionError("GPU not available!")

    if args.use_gpu:
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(123 + args.seed)
        torch.cuda.manual_seed_all(123 + args.seed)
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda", 0)
        ngpus = torch.cuda.device_count()
        print(f"{ngpus} GPUs available, but I only use one")
        print(torch.cuda.memory.list_gpu_processes())
    else:
        device = torch.device("cpu")
        print(f"Using cpu...")

    model_path = '%s/%s.py' % (args.dataset, args.model)
    if not os.path.exists(model_path):
        print('Please specify a valid dataset and a valid model.')
    model_path = '%s.%s' % (args.dataset, args.model)
    
    print('############################## %s ##############################' % model_path)
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, 'ClientModel')
    CustomDataset = getattr(mod, 'CustomDataset')
    calc_loss = getattr(mod, 'calc_loss')
    calc_pred = getattr(mod, 'calc_pred')

    tup = MAIN_PARAMS[args.dataset][args.t]
    num_rounds = args.num_rounds if args.num_rounds != -1 else tup[0]
    eval_every = args.eval_every if args.eval_every != -1 else tup[1]
    clients_per_round = args.clients_per_round if args.clients_per_round != -1 else tup[2]

    # Create 2 models
    model_params = MODEL_PARAMS[model_path]

    # Create client model, and share params with server model
    net = ClientModel(*model_params)
    if args.use_gpu:
        net = net.to(device)

    #criterion = nn.CrossEntropyLoss()
    #criterion = nn.CTCLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr)

    # Create server
    server = Server(net, device, args.server_lr)

    # Create clients
    clients = setup_clients(args.dataset, device, args.use_val_set, net, CustomDataset, calc_loss, calc_pred)
    client_ids, client_groups, client_num_samples = server.get_clients_info(clients)
    print('Clients in Total: %d' % len(clients))

    # Initial status
    print('--- Random Initialization ---')
    stat_writer_fn = get_stat_writer_function(client_ids, client_groups, client_num_samples, args)
    sys_writer_fn = get_sys_writer_function(args)
    #print_stats(0, server, clients, client_num_samples, args, stat_writer_fn, args.use_val_set)
    print("Done")

    # Simulate training
    for i in range(num_rounds):
        print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))

        # Select clients to train this round
        server.select_clients(i, online(clients), num_clients=clients_per_round)
        c_ids, c_groups, c_num_samples = server.get_clients_info(server.selected_clients)

        # Simulate server model training on selected clients' data
        sys_metrics = server.train_model(num_epochs=args.num_epochs, batch_size=args.batch_size, optimizer=optimizer, minibatch=args.minibatch)
        sys_writer_fn(i + 1, c_ids, sys_metrics, c_groups, c_num_samples)
        
        # Update server model
        server.update_model()

        # Test model
        if (i + 1) % eval_every == 0 or (i + 1) == num_rounds:
            print_stats(i + 1, server, clients, client_num_samples, args, stat_writer_fn, args.use_val_set)

            # Commenting out for faster testing
            #test_num = len(clients)
            #test_clients = random.sample(clients, test_num) 
            #sc_ids, sc_groups, sc_num_samples = server.get_clients_info(test_clients)
            #print('number of clients for test: {} of {} '.format(len(test_clients),len(clients)))
            #another_stat_writer_fn = get_stat_writer_function(sc_ids, sc_groups, sc_num_samples, args)
            # print_stats(i + 1, server, test_clients, client_num_samples, args, stat_writer_fn)
            #print_stats(i, server, test_clients, sc_num_samples, args, another_stat_writer_fn, "test")
    
def online(clients):
    """We assume all users are always online."""
    return clients


def create_clients(users, groups, train_data, test_data, model, device, dataset_fn, loss_fn, pred_fn):
    if len(groups) == 0:
        groups = [[] for _ in users]
    clients = [Client(u, g, train_data[u], test_data[u], model, device, dataset_fn, loss_fn, pred_fn) for u, g in zip(users, groups)]

    return clients


def setup_clients(dataset, device, use_val_set=False, model=None, dataset_fn=None, loss_fn=None, pred_fn=None):
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """
    eval_set = 'test' if not use_val_set else 'val'
    train_data_dir = os.path.join('/media/sf_vbox_shared/data/', dataset, 'data', 'train')
    test_data_dir = os.path.join('/media/sf_vbox_shared/data/', dataset, 'data', eval_set)

    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)

    clients = create_clients(users, groups, train_data, test_data, model, device, dataset_fn, loss_fn, pred_fn)

    return clients


def get_stat_writer_function(ids, groups, num_samples, args):

    def writer_fn(num_round, metrics, partition):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, partition, args.metrics_dir, '{}_{}'.format(args.metrics_name, 'stat'))

    return writer_fn


def get_sys_writer_function(args):

    def writer_fn(num_round, ids, metrics, groups, num_samples):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, 'train', args.metrics_dir, '{}_{}'.format(args.metrics_name, 'sys'))

    return writer_fn


def print_stats(
    num_round, server, clients, num_samples, args, writer, use_val_set):
    
    # Kiwan: Commenting this out because it is too slow (when you have 100s of client).
    #train_stat_metrics = server.test_model(clients, set_to_use='train')
    #print_metrics(train_stat_metrics, num_samples, prefix='train_')
    #writer(num_round, train_stat_metrics, 'train')

    eval_set = 'test' if not use_val_set else 'val'
    test_stat_metrics = server.test_model(clients, set_to_use=eval_set)
    # Print Total
    print_metrics(test_stat_metrics, num_samples, prefix='{}_'.format(eval_set))


def print_metrics(metrics, weights, prefix=''):
    """Prints weighted averages of the given metrics.

    Args:
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: dict with client ids as keys. Each entry is the weight
            for that client.
    """
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = metrics_writer.get_metrics_names(metrics)
    to_ret = None
    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        print('%s: %g, 5th: %g, 10th: %g, 25th: %g, 50th: %g, 75th: %g, 90th: %g, 95th: %g' \
              % (prefix + metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 5),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 25),
                 np.percentile(ordered_metric, 50),
                 np.percentile(ordered_metric, 75),
                 np.percentile(ordered_metric, 90),
                 np.percentile(ordered_metric, 95),
                 ))


if __name__ == '__main__':
    args = parse_args()
    main(args)
