import numpy as np
import copy
import torch

from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY

class Server:
    def __init__(self, client_model, device, lr):
        self.selected_clients = []
        self.updates = []
        self.client_model = client_model
        self.model = copy.deepcopy(client_model)
        self.device = device
        self.lr = lr

    def select_clients(self, my_round, possible_clients, num_clients=20):
        """Selects num_clients clients randomly from possible_clients.
        
        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:
            list of (num_train_samples, num_test_samples)
        """

        num_clients = int(min(num_clients, len(possible_clients)))
        np.random.seed(my_round)

        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]

    def train_model(self, num_epochs=1, batch_size=10, optimizer=None, minibatch=None, clients=None):
        """Trains self.model on given clients.
        
        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs
        and batches.

        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            bytes_written: number of bytes written by each client to server 
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """
        if clients is None:
            clients = self.selected_clients
        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0} for c in clients}

        for c in clients:
            c._model.load_state_dict(self.model.state_dict())
            comp, num_samples, update = c.train(num_epochs, batch_size, optimizer, minibatch)

            #sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
            #sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.size
            sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp


            #if compression == "channel_topk":
            #for name in update:
            #    if "weight" in name:
            #        mask = (abs(update[name]) >= abs(update[name]).flatten().topk(int((update[name].flatten().shape[0]) * (1 - sparsity)))[0][-1])
            #        update[name] *= mask
            self.updates.append((num_samples, update))

        return sys_metrics

    def update_model(self):
        accum = {}
        for name, param in self.model.state_dict().items():
            accum[name] = torch.zeros(param.shape, dtype=param.dtype).to(self.device)

        total_weight = 0
        for (client_samples, update) in self.updates:
            total_weight += client_samples
            for name in accum:
                accum[name] += update[name] * client_samples
        for name in accum:
            # TODO: What is the right way to handle BN? Currently, I just average everything.
            if "num_batches_tracked" in name:
                accum[name] //= total_weight # must be Int
            else:
                accum[name] /= total_weight

        with torch.no_grad():
            for name, param in self.model.state_dict().items():
                if "num_batches_tracked" in name:
                    param += (self.lr * accum[name]).long()
                else:
                    param += self.lr * accum[name]
        self.updates = []

    def test_model(self, clients_to_test, set_to_use='test'):
        """Tests self.model on given clients.

        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
            set_to_use: dataset to test on. Should be in ['train', 'test'].
        """
        metrics = {}
        _metrics = {}

        if clients_to_test is None:
            clients_to_test = self.selected_clients

        for client in clients_to_test:
            client._model.load_state_dict(self.model.state_dict())
            c_metrics = client.test(set_to_use)
            metrics[client.id] = c_metrics
        
        return metrics

    def get_clients_info(self, clients):
        """Returns the ids, hierarchies and num_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.selected_clients

        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_samples = {c.id: c.num_samples for c in clients}
        return ids, groups, num_samples
