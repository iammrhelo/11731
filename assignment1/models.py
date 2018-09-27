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

        # Build layers
        self.embed = nn.Embedding(self.num_embeddings, self.embed_size)
        self.rnn = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers,
                           dropout=self.dropout_rate)
        self.h2o = nn.Linear(self.hidden_size, self.num_embeddings)

    def forward(self, x, hidden=None, src_encodings=None):
        embed_x = self.embed(x)
        rnn_output, hidden = self.rnn(embed_x, hidden)
        output = self.h2o(F.relu(rnn_output))
        return output, hidden


class LuongDecoder(nn.Module):
    """
    An attention based decoder that also takes in encoder_outputs
    """

    def __init__(self, opt):
        super(LuongDecoder, self).__init__()
        self.num_embeddings = opt["num_embeddings"]
        self.embed_size = opt["embed_size"]
        self.hidden_size = opt["hidden_size"]
        self.num_layers = opt["num_layers"]
        self.dropout_rate = opt["dropout_rate"]

        # Build layers
        self.embed = nn.Embedding(self.num_embeddings, self.embed_size)
        self.rnn = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers,
                           dropout=self.dropout_rate)
        self.h2o = nn.Linear(2 * self.hidden_size, self.num_embeddings)

        # Use concat attention for now
        attn_opt = {'hidden_size': self.hidden_size}
        self.attn = ConcatAttention(attn_opt)

    def forward(self, x, hidden=None, src_encodings=None):
        """
        Args:
            x : decoder input
            src_encodings : encoder states
            hidden : decoder hidden state
        """
        embed_x = self.embed(x)
        rnn_output, hidden = self.rnn(embed_x, hidden)

        attn_weights = self.attn.forward(rnn_output, src_encodings)

        # (1, batch_size, hidden_size)
        context = (attn_weights * src_encodings).sum(dim=0)
        context = context.unsqueeze(dim=0)

        hidden_plus_context = torch.cat([rnn_output, context], dim=-1)
        output = self.h2o(hidden_plus_context)
        return output, hidden


class ConcatAttention(nn.Module):
    """
    Score function: v_a^T * tanh(W_a [ h_t, h_s ])
    """

    def __init__(self, opt):
        super(ConcatAttention, self).__init__()
        self.hidden_size = opt["hidden_size"]

        self.Wa = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.Va = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self, h_t, src_encodings):
        """
        Args:
            h_t : (lenght, batch_size, hidden_size)
            src_encodings : (src_length, batch_size, hidden_size)
        Returns:
            attn_weights: (src_length, batch_size ) 
        """
        src_length, _, _ = src_encodings.shape

        repeat_ht = h_t.repeat(src_length, 1, 1)
        # (src_length, batch_size, 2 * hidden_size)
        concat_hidden = torch.cat([repeat_ht, src_encodings], dim=-1)
        # (src_length, batch_size)
        scores = self.Va(torch.tanh(self.Wa(concat_hidden)))
        attn_weights = F.softmax(scores, dim=0)
        return attn_weights
