import random
import warnings
import torch
import torch.nn as nn
from baseline_constants import ACCURACY_KEY

class Client:
    def __init__(self, client_id, group=None, train_data={'x' : [],'y' : []}, eval_data={'x' : [],'y' : []}, model=None, device=torch.device("cpu"), dataset_fn=None, loss_fn=None, pred_fn=None):
        self.id = client_id
        self.group = group
        self.train_data = train_data
        self.eval_data = eval_data
        self._model = model
        self.device = device
        self.dataset_fn = dataset_fn
        self.loss_fn = loss_fn
        self.pred_fn = pred_fn

    def train(self, num_epochs, batch_size, optimizer, minibatch=None):
        """Trains on self.model using the client's train_data.

        Args:
            num_epochs: Number of epochs to train. Unsupported if minibatch is provided (minibatch has only 1 epoch)
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            comp: number of FLOPs executed in training process
            num_samples: number of samples used in training
            update: set of weights
            update_size: number of bytes in update
        """

        if minibatch is None:
            data = self.train_data
        else:
            frac = min(1.0, minibatch)
            num_data = max(1, int(frac*len(self.train_data["x"])))
            xs, ys = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data))
            data = {'x': xs, 'y': ys}

            # Minibatch trains for only 1 epoch - multiple local epochs don't make sense!
            num_epochs = 1

        dataset = self.dataset_fn(data)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                shuffle=True, num_workers=0)

        # To calculate each gradient
        grad = {}
        self._model.train()
        for name, param in self._model.state_dict().items():
            grad[name] = -param.data.clone().detach()

        total_loss = 0
        total_acc = 0
        total = 0
        for _ in range(num_epochs):
            for (x, y) in train_loader:
                x = x.to(self.device)
                optimizer.zero_grad()
                out = self._model(x)
                y = y.type(torch.LongTensor).to(self.device)
                loss = self.loss_fn(out, y)
                total += y.shape[0]
                total_loss += loss.detach().cpu().numpy() * y.shape[0]
                predicted = self.pred_fn(out)
                acc = ((predicted == y).sum().item())
                total_acc += acc
                loss.backward()
                optimizer.step()

        total_loss /= total
        total_acc /= total

        for name, param in self._model.state_dict().items():
            grad[name] += param.data.clone().detach()

        num_train_samples = len(data['y'])

        return None, num_train_samples, grad


    def test(self, set_to_use='test'):
        """Tests self.model on self.test_data.
        
        Args:
            set_to_use. Set to test on. Should be in ['train', 'test'].
        Return:
            dict of metrics returned by the model.
        """
        assert set_to_use in ['train', 'test', 'val']
        if set_to_use == 'train':
            data = self.train_data
        elif set_to_use == 'test' or set_to_use == 'val':
            data = self.eval_data

        dataset = self.dataset_fn(data)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=10,
                                                shuffle=True, num_workers=0)

        self._model.eval()
        loss = 0
        acc = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                out = self._model(x)
                y = y.type(torch.LongTensor).to(self.device)
                loss += self.loss_fn(out, y) * y.shape[0]
                predicted = self.pred_fn(out)
                acc += ((predicted == y).sum().item())
                #if ((predicted == y).sum().item()) > 0.3 * y.shape[0]:
                #    print(((predicted == y).sum().item()), predicted, y)
                total += y.flatten().shape[0]
        loss /= total
        acc /= total

        return {ACCURACY_KEY: acc, 'loss': loss.cpu()}

    @property
    def num_test_samples(self):
        """Number of test samples for this client.

        Return:
            int: Number of test samples for this client
        """
        if self.eval_data is None:
            return 0
        return len(self.eval_data['y'])

    @property
    def num_train_samples(self):
        """Number of train samples for this client.

        Return:
            int: Number of train samples for this client
        """
        if self.train_data is None:
            return 0
        return len(self.train_data['y'])

    @property
    def num_samples(self):
        """Number samples for this client.

        Return:
            int: Number of samples for this client
        """
        train_size = 0
        if self.train_data is not None:
            train_size = len(self.train_data['y'])

        test_size = 0 
        if self.eval_data is not  None:
            test_size = len(self.eval_data['y'])
        return train_size + test_size

    @property
    def model(self):
        """Returns this client reference to model being trained"""
        return self._model

    @model.setter
    def model(self, model):
        warnings.warn('The current implementation shares the model among all clients.'
                      'Setting it on one client will effectively modify all clients.')
        self._model = model
