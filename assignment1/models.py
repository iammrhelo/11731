import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    A basic encoder implementation with options
    """

    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.num_embeddings = opt["num_embeddings"]
        self.embed_size = opt["embed_size"]
        self.hidden_size = opt["hidden_size"]
        self.num_layers = opt["num_layers"]
        self.dropout_rate = opt["dropout_rate"]
        self.bidirectional = opt["bidirectional"]

        # Build n
        self.embed = nn.Embedding(self.num_embeddings, self.embed_size)

        self.rnn = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers,
                           dropout=self.dropout_rate, bidirectional=self.bidirectional)

    def forward(self, x, hidden=None):
        embed_x = self.embed(x)
        output, hidden = self.rnn(embed_x, hidden)
        return output, hidden


class Decoder(nn.Module):
    """
    A basic decoder implemenation with options
    """

    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.num_embeddings = opt["num_embeddings"]
        self.embed_size = opt["embed_size"]
        self.hidden_size = opt["hidden_size"]
        self.num_layers = opt["num_layers"]
        self.dropout_rate = opt["dropout_rate"]

        # Build n
        self.embed = nn.Embedding(self.num_embeddings, self.embed_size)
        self.rnn = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers,
                           dropout=self.dropout_rate)
        self.out = nn.Linear(self.hidden_size, self.num_embeddings)

    def forward(self, x, hidden=None):
        embed_x = self.embed(x)
        rnn_output, hidden = self.rnn(embed_x, hidden)
        output = self.out(F.relu(rnn_output))
        return output, hidden


class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.input_size = opt["input_size"]
        self.hidden_size = opt["hidden_size"]

    def forward(self, x):
        raise NotImplementedError
