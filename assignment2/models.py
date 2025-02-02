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

        self.rnn = nn.LSTM(
            self.embed_size,
            self.hidden_size,
            self.num_layers,
            dropout=self.dropout_rate,
            bidirectional=self.bidirectional)

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


def lstm_cell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    # type: (Tensor, Tuple[Tensor, Tensor], Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    hx, cx = hidden

    gates = torch.mm(input, w_ih.t()) + torch.mm(hx, w_hh.t()) + b_ih + b_hh

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy

class HyperEncoder(nn.Module):
    """
    A basic encoder implementation with options
    """

    def __init__(self, opt):
        super(HyperEncoder, self).__init__()
        self.num_embeddings = opt["num_embeddings"]
        self.embed_size = opt["embed_size"]
        self.hidden_size = opt["hidden_size"]
        self.num_layers = opt["num_layers"]
        self.dropout_rate = opt["dropout_rate"]
        self.bidirectional = opt["bidirectional"]
        self.lang_embed_size = opt["lang_embedding_size"]

        # Build n
        self.embed = nn.Embedding(self.num_embeddings, self.embed_size)

        self.c_init = torch.nn.Parameter(torch.zeros(1, self.hidden_size))
        self.h_init = torch.nn.Parameter(torch.zeros(1, self.hidden_size))

        if self.bidirectional:
            self.c_init_reverse = torch.nn.Parameter(torch.zeros(1, self.hidden_size))
            self.h_init_reverse = torch.nn.Parameter(torch.zeros(1, self.hidden_size))

        self.hyper_w_ih_l0 = nn.Linear(self.lang_embed_size, 4*self.hidden_size*self.embed_size)
        self.hyper_w_hh_l0 = nn.Linear(self.lang_embed_size, 4*self.hidden_size*self.hidden_size)
        self.hyper_b_ih_l0 = nn.Linear(self.lang_embed_size, 4*self.hidden_size)
        self.hyper_b_hh_l0 = nn.Linear(self.lang_embed_size, 4*self.hidden_size)
        if self.bidirectional:
            self.hyper_w_ih_l0_reverse = nn.Linear(self.lang_embed_size, 4*self.hidden_size*self.embed_size)
            self.hyper_w_hh_l0_reverse = nn.Linear(self.lang_embed_size, 4*self.hidden_size*self.hidden_size)
            self.hyper_b_ih_l0_reverse = nn.Linear(self.lang_embed_size, 4*self.hidden_size)
            self.hyper_b_hh_l0_reverse = nn.Linear(self.lang_embed_size, 4*self.hidden_size)


        self.rnn = nn.LSTM(
            self.embed_size,
            self.hidden_size,
            self.num_layers,
            dropout=self.dropout_rate,
            bidirectional=self.bidirectional)



    def hyper_rnn_forward(self, input, lengths, batch_lang):
        input = input.permute(1,0,2)
        batch_size = input.size()[0]
        # print("batch_lang", batch_lang, batch_lang.size())
        w_ih = self.hyper_w_ih_l0(batch_lang).view(4*self.hidden_size, -1)
        w_hh = self.hyper_w_hh_l0(batch_lang).view(4*self.hidden_size, -1)
        b_ih = self.hyper_b_ih_l0(batch_lang).view(4*self.hidden_size)
        b_hh = self.hyper_b_hh_l0(batch_lang).view(4*self.hidden_size)

        w_ih_reverse = self.hyper_w_ih_l0_reverse(batch_lang).view(4*self.hidden_size, -1)
        w_hh_reverse = self.hyper_w_hh_l0_reverse(batch_lang).view(4*self.hidden_size, -1)
        b_ih_reverse = self.hyper_b_ih_l0_reverse(batch_lang).view(4*self.hidden_size)
        b_hh_reverse = self.hyper_b_hh_l0_reverse(batch_lang).view(4*self.hidden_size)


        max_seq_len = input.size()[1]

        output_forward = []
        output_reverse = []

        hy_forward, cy_forward = self.h_init, self.c_init
        hy_reverse, cy_reverse = self.h_init_reverse, self.c_init_reverse
        output_cell_forward = []
        for i in range(max_seq_len):
            hy_forward, cy_forward = lstm_cell(input[:,i,:], (hy_forward, cy_forward), w_ih, w_hh, b_ih, b_hh)

            hy_reverse, cy_reverse = lstm_cell(input[:,max_seq_len-i-1,:], (hy_reverse, cy_reverse), w_ih_reverse, w_hh_reverse, b_ih_reverse, b_hh_reverse)
            output_forward.append(hy_forward)
            output_reverse.append(hy_reverse)
            output_cell_forward.append(cy_forward)

        output_forward_hyper = torch.stack(output_forward, dim=1)
        output_reverse_hyper = torch.stack(output_reverse, dim=1)

        output_cell_forward = torch.stack(output_cell_forward, dim=1)


        final_forward = []
        final_reverse = []
        final_cell_forward = []
        for i in range(batch_size):
            final_forward.append(output_forward_hyper[i, lengths[i]-1,:])
            final_cell_forward.append(output_cell_forward[i, lengths[i]-1,:])

        #     final_reverse.append(output_reverse_hyper[i, lengths[i]-1,:])

        final_forward = torch.stack(final_forward, dim=0)
        final_cell_forward = torch.stack(final_cell_forward, dim=0)
        # final_reverse = torch.stack(final_reverse, dim=0)


        final_output = torch.cat((output_forward_hyper, output_reverse_hyper), dim=2)

        return final_output.permute(1,0,2), (final_forward.unsqueeze(dim=0), final_cell_forward.unsqueeze(dim=0))

    def forward(self, x, lengths, batch_lang, hidden=None):
        embed_x = self.embed(x)
        # hidden should corresponding to the hidden state at lengths

        # HYPER = False
        HYPER = True

        if HYPER:
            output, hidden = self.hyper_rnn_forward(embed_x, lengths, batch_lang)

        else:
            packed_embed_x = pack_padded_sequence(embed_x, lengths)
            packed_output, hidden = self.rnn(packed_embed_x, hidden)
            output, _ = pad_packed_sequence(packed_output)

            if self.bidirectional:
                # We only give the forward hidden states here
                def get_forward_op(h):
                    _, batch_size, hidden_size = h.shape
                    sep_h = h.view(self.num_layers, 2, batch_size, hidden_size)
                    return sep_h[:, 0, :, :]

                hidden = tuple(map(get_forward_op, hidden))

        # print(output.size(), "hidden", hidden[0].size(), hidden[1].size())#torch.Size([69, 64, 1024]) hidden torch.Size([1, 64, 512]) torch.Size([1, 64, 512])
        # fooo
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
        self.rnn = nn.LSTM(
            self.embed_size,
            self.hidden_size,
            self.num_layers,
            dropout=self.dropout_rate)
        self.h2o = nn.Linear(self.hidden_size, self.num_embeddings)

    def forward(self, x, hidden=None, src_encodings=None, tgt_tensor=None):
        embed_x = self.embed(x)
        rnn_output, hidden = self.rnn(embed_x, hidden)
        output = self.h2o(rnn_output)
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

    Currently, we use BiLinear attention
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
