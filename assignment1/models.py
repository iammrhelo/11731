import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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

        self.rnn = nn.LSTM(self.embed_size,
                           self.hidden_size,
                           self.num_layers,
                           dropout=self.dropout_rate,
                           bidirectional=self.bidirectional)

    def forward(self, x, lengths, hidden=None):
        embed_x = self.embed(x)
        packed_embed_x = pack_padded_sequence(embed_x, lengths)
        packed_output, hidden = self.rnn(packed_embed_x, hidden)
        output, _ = pad_packed_sequence(packed_output)

        if self.bidirectional:
            # Handle bidirectional stuff here
            def concat_op(h):
                return torch.cat([h[:1], h[1:2]], dim=-1)

            hidden = tuple(map(concat_op, hidden))
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

    def forward(self, x, hidden=None, src_encodings=None, tgt_tensor=None):
        embed_x = self.embed(x)
        rnn_output, hidden = self.rnn(embed_x, hidden)
        output = self.h2o(rnn_output)
        return output, hidden


class DenseBridge(nn.Module):
    def __init__(self, opt):
        super(DenseBridge, self).__init__()
        self.input_size = opt["input_size"]
        self.output_size = opt["output_size"]
        self.dropout_rate = opt["dropout_rate"]

        self.fc = nn.Linear(self.input_size, self.output_size)

    def forward(self, encoder_hidden):
        """
        Maps encoder hidden states to decoder
        Apply dropout layer followed by tanh
        """
        raise NotImplementedError


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

        # Use concatenate attention for now
        attn_opt = {'input_size': 2 * self.hidden_size,
                    "output_size": self.hidden_size}
        self.attn = ConcatAttention(attn_opt)

    def forward(self, x, hidden=None, src_encodings=None, tgt=None):
        """
        Args:
            x: decoder input (length, batch_size)
            decoder_hidden : decoder hidden state 
            src_encodings : encoder states (src_length, batch_size, hidden_size)
            tgt : target output tensor (length, batch_size)
        """
        if tgt is None:
            tgt_length = 1
        else:
            tgt_length, _ = tgt.shape

        # Feed ground truth for now
        output_list = []
        for step in range(tgt_length):
            step_input = x[step:step+1]
            step_output, hidden = self.forward_step(
                step_input, hidden, src_encodings)
            output_list.append(step_output)

        output = torch.cat(output_list)

        return output, hidden

    def forward_step(self, x, hidden=None, src_encodings=None):
        """
        Perform a single step of the decoder
        Length dimension should be 1
        Args:
            x:              step input (1, batch_size)
            hidden:         decoder hidden
            src_encodings:  hidden states from encoder            
        """
        embed_x = self.embed(x)
        rnn_output, hidden = self.rnn(embed_x, hidden)

        # Compute context vector
        attn_weights = self.attn.forward(rnn_output, src_encodings)
        context = (attn_weights * src_encodings).sum(dim=0).unsqueeze(0)

        # Predict final output
        hidden_plus_context = torch.cat([context, rnn_output], dim=-1)
        output = self.h2o(hidden_plus_context)
        return output, hidden


class ConcatAttention(nn.Module):
    """
    Score function: v_a^T * tanh(W_a [ h_t, h_s ])
    """

    def __init__(self, opt):
        super(ConcatAttention, self).__init__()
        self.input_size = opt["input_size"]
        self.output_size = opt["output_size"]

        self.Wa = nn.Linear(self.input_size, self.output_size)
        self.Va = nn.Linear(self.output_size, 1)

    def forward(self, h_t, src_encodings):
        """
        Args:
            h_t : (lenght, batch_size, hidden_size)
            src_encodings : (src_length, batch_size, hidden_size)
        Returns:
            attn_weights: (src_length, batch_size ) 
        """
        src_length, _, _ = src_encodings.shape
        #tgt_length, _, _ = h_t.shape

        repeat_ht = h_t.repeat(src_length, 1, 1)
        concat_hidden = torch.cat([repeat_ht, src_encodings], dim=-1)
        # (tgt_length, src_length, batch_size, hidden_size )
        scores = self.Va(torch.tanh(self.Wa(concat_hidden)))
        attn_weights = F.softmax(scores, dim=0)
        return attn_weights
