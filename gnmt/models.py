import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CPG(nn.Module):
    """
    Contextual Parameter Generator for Domain Adaptation
    Generators weights for Encoder LSTM and Decoder LSTM 
    """

    def __init__(self, opt):
        super(CPG, self).__init__()
        # Architecture Params
        self.num_domains = opt["num_domains"]
        self.domain_embed_size = opt["domain_embed_size"]
        self.relevant_information_size = opt["relevant_information_size"]
        self.encoder_rnn_param_size = opt["encoder_rnn_param_size"]
        self.decoder_rnn_param_size = opt["decoder_rnn_param_size"]

        # Domain Embedding
        self.embed = nn.Embedding(self.num_domains, self.domain_embed_size)

        # Encoder
        self.P_enc = nn.Linear(self.domain_embed_size,
                               self.relevant_information_size)
        self.W_enc = nn.Linear(
            self.relevant_information_size, self.encoder_rnn_param_size)

        # Decoder
        self.P_dec = nn.Linear(self.domain_embed_size,
                               self.relevant_information_size)
        self.W_dec = nn.Linear(
            self.relevant_information_size, self.decoder_rnn_param_size)

    def forward(self, d):
        """
        Generates parameters(theta) for encoder, attention and decoder
        Weights are generated for a single batch, therefore all the data 
        in the batch should be in the same domain
        """
        assert d.dim() == 1

        embed_d = self.embed(d)

        theta_enc = self.W_enc(self.P_enc(embed_d))
        theta_dec = self.W_dec(self.P_dec(embed_d))

        return theta_enc, theta_dec


class Encoder(nn.Module):
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

        self.rnn = nn.LSTM(
            self.embed_size,
            self.hidden_size,
            self.num_layers,
            dropout=self.dropout_rate,
            bidirectional=self.bidirectional)

    def get_rnn_parameter_size(self):
        """
        Get size of rnn parameters
        """
        size = 0
        for param in self.rnn.parameters():
            size += param.numel()
        return size

    def set_rnn_parameters(self, W_flat):
        """
        Sets model parameters with a Flat matrix coming from CPG
        Need to process Flat matrix into groups
        """
        assert W_flat.shape[0] == 1
        W_flat = W_flat[0]

        # Indicating start of chunk
        ptr = 0
        for weights in self.rnn._all_weights:
            for weight_name in weights:
                orig_weight = getattr(self.rnn, weight_name)
                numel = orig_weight.numel()

                # Convert CPG chunk to RNN weight
                chunk = W_flat[ptr:ptr+numel]
                ptr += numel
                weight = nn.Parameter(chunk.view(orig_weight.shape))
                # Set to rnn weights
                setattr(self.rnn, weight_name, weight)

        assert ptr == W_flat.shape[0]

    def forward(self, x, lengths, hidden=None):
        embed_x = self.embed(x)
        packed_embed_x = pack_padded_sequence(embed_x, lengths)
        # hidden should corresponding to the hidden state at lengths
        packed_output, hidden = self.rnn(packed_embed_x, hidden)
        output, _ = pad_packed_sequence(packed_output)

        if self.bidirectional:
            # We only give the forward hidden states here
            def get_forward_op(h):
                _, batch_size, hidden_size = h.shape
                sep_h = h.view(self.num_layers, 2, batch_size, hidden_size)
                return sep_h[:, 0, :, :]

            hidden = tuple(map(get_forward_op, hidden))

        return output, hidden


class LuongDecoder(nn.Module):
    """
    An attention based decoder that also takes in encoder_outputs
    """

    def __init__(self, opt, attn):
        super(LuongDecoder, self).__init__()
        self.num_embeddings = opt["num_embeddings"]
        self.embed_size = opt["embed_size"]
        self.hidden_size = opt["hidden_size"]
        self.num_layers = opt["num_layers"]
        self.dropout_rate = opt["dropout_rate"]

        # Build layers
        self.embed = nn.Embedding(self.num_embeddings, self.embed_size)
        self.rnn = nn.LSTM(
            self.embed_size,
            self.hidden_size,
            self.num_layers,
            dropout=self.dropout_rate)

        # Global Attention: Concat scoring function
        self.attn = attn

        # Final output prediction
        if self.attn.attn_type == "Concat":
            concat_size = self.attn.input_size
        else:
            concat_size = self.attn.input_size + self.attn.output_size

        # attentional layer
        self.c2h = nn.Linear(concat_size, self.hidden_size)
        # output layer
        self.h2o = nn.Linear(self.hidden_size, self.num_embeddings)

    def get_rnn_parameter_size(self):
        """
        Get size of rnn parameters
        """
        size = 0
        for param in self.rnn.parameters():
            size += param.numel()
        return size

    def set_rnn_parameters(self, W_flat):
        """
        Sets model parameters with a Flat matrix coming from CPG
        Need to process Flat matrix into groups
        """
        assert W_flat.shape[0] == 1
        W_flat = W_flat[0]

        # Indicating start of chunk
        ptr = 0
        for weights in self.rnn._all_weights:
            for weight_name in weights:
                orig_weight = getattr(self.rnn, weight_name)
                numel = orig_weight.numel()

                # Convert CPG chunk to RNN weight
                chunk = W_flat[ptr:ptr+numel]
                ptr += numel
                weight = nn.Parameter(chunk.view(orig_weight.shape))
                # Set to rnn weights
                setattr(self.rnn, weight_name, weight)

        assert ptr == W_flat.shape[0]

    def forward(self,
                x,
                hidden=None,
                src_encodings=None,
                src_lengths=None,
                tgt=None):
        """
        Args:
            x: decoder input (length, batch_size)
            decoder_hidden : decoder hidden state
            src_encodings : encoder states (src_length, batch_size, hidden_size)
            src_lengths: corresponding lengths of source encoder
            tgt : target output tensor (length, batch_size)
        """
        if tgt is None:
            tgt_length = 1
        else:
            tgt_length, _ = tgt.shape

        # Feed ground truth for now
        embed_x = self.embed(x)
        # rnn_output: (tgt_length, batch_size, hidden_size)
        hidden = [h.contiguous() for h in hidden]
        rnn_output, hidden = self.rnn.forward(embed_x, hidden)

        # attn_weights: (src_length, tgt_length, batch_size, hidden_size)
        # src_encodings: (src_length, batch_size, hidden_size)
        context = self.attn.forward(rnn_output, src_encodings, src_lengths)
        concat = torch.cat([context, rnn_output], dim=-1)

        # attentional layer
        attn_hidden_state = torch.tanh(
            self.c2h(F.dropout(concat, self.dropout_rate)))

        # output logits
        output = self.h2o(F.dropout(attn_hidden_state, self.dropout_rate))

        return output, hidden


class GlobalAttention(nn.Module):
    """
    TODO: Implement 3 types of attention:
    1. Dot
    2. General: h_t^T W_a * h_s
    3. Concat: v_a^T * tanh(W_a^T[ h_t, h_s ])
    """

    def __init__(self, attn_type, mask_attn, encoder_hidden_size,
                 decoder_hidden_size):
        super(GlobalAttention, self).__init__()
        self.attn_type = attn_type
        self.mask_attn = mask_attn
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        if self.attn_type == "Concat":
            self.input_size = self.encoder_hidden_size + self.decoder_hidden_size
            self.output_size = self.decoder_hidden_size
            self.Wa = nn.Linear(self.input_size, self.output_size, bias=False)
            self.Va = nn.Linear(self.output_size, 1, bias=False)
        elif self.attn_type == "General":
            self.input_size = self.encoder_hidden_size
            self.output_size = self.decoder_hidden_size
            self.Wa = nn.Linear(self.input_size, self.output_size, bias=False)
        else:
            raise ValueError("Unknown attn_type: {}".format(self.attn_type))

    def forward(self, h_t, src_encodings, src_lengths=None):
        """
        Gets context vector here
        Args:
            h_t : (length, batch_size, hidden_size)
            src_encodings : (src_length, batch_size, hidden_size)
        Returns:
            attn_weights: (src_length, batch_size, 1)
        """
        src_length, batch_size, src_dim = src_encodings.shape
        tgt_length, batch_size, tgt_dim = h_t.shape

        # Mask src_encodings with -inf

        # First we formulate into 4D array
        # (src_length, tgt_length, batch_size, hidden_size)
        if self.attn_type.lower() == "Concat".lower():
            expand_ht = h_t.expand(src_length, -1, -1, -1)
            expand_src_encodings = src_encodings.expand(
                tgt_length, -1, -1, -1).permute(1, 0, 2, 3)
            concat_hidden = torch.cat(
                [expand_ht, expand_src_encodings], dim=-1)
            scores = self.Va(torch.tanh(self.Wa(concat_hidden)))

        elif self.attn_type.lower() == "General".lower():
            # Wa * h_s: (src_length, batch_size, tgt_dim)
            Wahs = self.Wa(src_encodings)
            # batch first for torch.bmm
            bf_Wahs = Wahs.permute(1, 0, 2)
            bf_h_t = h_t.permute(1, 2, 0)
            expand_src_encodings = src_encodings.expand(
                tgt_length, -1, -1, -1).permute(1, 0, 2, 3)
            # scores: (batch_size, src_length, target_length)
            scores = torch.bmm(bf_Wahs, bf_h_t)
            scores = scores.permute(1, 2, 0).unsqueeze(-1)

        else:
            raise ValueError("Unknown attn_type: {}".format(self.attn_type))

        if self.mask_attn:
            # create mask
            mask = torch.zeros(
                src_length, batch_size, dtype=src_encodings.dtype)
            if scores.is_cuda:
                mask = mask.cuda()
            for idx, length in enumerate(src_lengths):
                mask[:length, idx] = 1

            mask = mask.view(src_length, 1, batch_size, 1)
            # masked softmax, haha
            attn_weights = masked_softmax(scores, mask, dim=0)
        else:
            attn_weights = F.softmax(scores, dim=0)
        context = (attn_weights * expand_src_encodings).sum(dim=0)
        return context


def masked_softmax(scores, mask, dim=0):
    """
    scores should have the same shape as mask
    """
    scores_exp = torch.exp(scores)
    masked_scores_exp = scores_exp * mask
    masked_scores_softmax = masked_scores_exp / \
        torch.sum(masked_scores_exp, dim=0, keepdim=True)
    return masked_scores_softmax


if __name__ == "__main__":

    encoder_opt = {
        "num_embeddings": 3,
        "embed_size": 5,
        "hidden_size": 7,
        "num_layers": 2,
        "dropout_rate": 0.2,
        "bidirectional": True
    }

    encoder = Encoder(encoder_opt)
    encoder_rnn_param_size = encoder.get_rnn_parameter_size()
    print("encoder rnn param size", encoder_rnn_param_size)

    opt = {
        "num_domains": 3,
        "domain_embedding_size": 10,
        "relevant_information_size": 5,
        "encoder_rnn_param_size": encoder_rnn_param_size,
        "attn_param_size": 20,
        "decoder_rnn_param_size": 100
    }
    cpg = CPG(opt)
    d = torch.LongTensor([1])
    theta_enc, _, _ = cpg.forward(d)

    encoder.set_rnn_parameters(theta_enc)
    encoder.parameters()
