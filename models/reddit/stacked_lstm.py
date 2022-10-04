import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pickle
import collections

VOCABULARY_PATH = '/media/sf_vbox_shared/data/reddit/vocab/reddit_vocab.pck'

def calc_loss(out, y):
    out = torch.swapaxes(out, 1, 2)
    #return nn.CTCLoss(zero_infinity=True)(out, y, torch.full(size=(out.shape[1],), fill_value=out.shape[0], dtype=torch.long), torch.full(size=(y.shape[0],), fill_value=y.shape[1], dtype=torch.long))
    #print(out.shape)
    #print(y.shape)
    return nn.CrossEntropyLoss()(out, y)

unk_symbol = 1
pad_symbol = 0

def calc_pred(out):
    #pred = torch.argmax(out.data, -1)
    #_, _, unk_symbol, pad_symbol = load_vocab()
    _, pred = torch.max(out.data, -1)
    pred[pred==unk_symbol] = -1
    pred[pred==pad_symbol] = -1

    return pred

def load_vocab():
    vocab_file = pickle.load(open(VOCABULARY_PATH, 'rb'))
    vocab = collections.defaultdict(lambda: vocab_file['unk_symbol'])
    vocab.update(vocab_file['vocab'])

    return vocab, vocab_file['size'], vocab_file['unk_symbol'], vocab_file['pad_symbol']

class ClientModel(nn.Module):
    def __init__(self, seq_len, n_hidden, num_layers,
            keep_prob=1.0, max_grad_norm=5, init_scale=0.1):
        super(ClientModel, self).__init__()

        self.vocab, self.vocab_size, self.unk_symbol, self.pad_symbol = load_vocab()

        n_emb = n_hidden

        self.emb = nn.Embedding(self.vocab_size, n_emb)
        self.lstm = nn.LSTM(n_emb, n_hidden, num_layers, dropout=1-keep_prob, batch_first=True)
        self.fc = nn.Linear(n_hidden, self.vocab_size)

    def forward(self, x):
        #x = self.emb(x.view(len(x), len(x[0]), -1).type(torch.LongTensor))
        x = self.emb(x)
        x, state = self.lstm(x)
        x = self.fc(x)

        return x


class CustomDataset(Dataset):
    def __init__(self, data):
        self.vocab, self.vocab_size, self.unk_symbol, self.pad_symbol = load_vocab()
        data_x = data['x']
        data_y = data['y']

        data_x_new = []
        data_y_new = []
        for c, l in zip(data_x, data_y):
            data_x_new.extend(c)
            data_y_new.extend(l["target_tokens"])

        for i, sentence in enumerate(data_x_new):
            #data_x_new[i] = self._tokens_to_ids([s for s in sentence])
            data_x_new[i] = [self.vocab[x] for x in sentence]

        for i, sentence in enumerate(data_y_new):
            #data_y_new[i] = self._tokens_to_ids([s for s in sentence])
            data_y_new[i] = [self.vocab[y] for y in sentence]

        self.data_x = [torch.Tensor(x).type(torch.LongTensor) for x in data_x_new]
        self.data_y = [torch.Tensor(y) for y in data_y_new]
        #self.data_x = data_x_new
        #self.data_y = data_y_new

    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, i):
        return self.data_x[i], self.data_y[i]
