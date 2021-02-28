import torch.nn as nn
from src.utils.utils import *
from src.models.FARNN import IntentIntegrateSaperate_B, IntentIntegrateSaperateBidirection_B,\
    FSARNNIntegrateEmptyStateSaperateGRU

class MarryupRe2rnn(nn.Module):
    def __init__(self, pretrained_embed, config=None, label_size=None, trans_r_1=None, trans_r_2=None, embed_r=None, trans_wildcard=None, h1_forward=None):
        super(MarryupRe2rnn, self).__init__()

        self.pretrained_embed = pretrained_embed
        self.config = config
        self.RE_D = config.re_tag_dim
        self.label_size = label_size
        self.bidirection = bool(config.bidirection)
        self.is_cuda = torch.cuda.is_available()
        self.V, self.D = pretrained_embed.shape

        self.trans_r_1 = trans_r_1
        self.trans_r_2 = trans_r_2
        self.embed_r = embed_r
        self.trans_wildcard = trans_wildcard
        self.h1_forward = h1_forward
        #prepare IntentIntegrateSaperate_B
        self.fsa = IntentIntegrateSaperate_B(self.pretrained_embed, self.config, self.trans_r_1, self.trans_r_2, self.embed_r,
                                             self.trans_wildcard, self.h1_forward)

        # merry up

        if config.rnn == 'RNN':
            self.rnn = nn.RNN(input_size=config.embed_dim,
                              hidden_size=config.rnn_hidden_dim,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=self.bidirection,
                              bias=False)

        elif config.rnn == 'LSTM':
            self.rnn = nn.LSTM(input_size=config.embed_dim,
                               hidden_size=config.rnn_hidden_dim,
                               num_layers=1,
                               batch_first=True,
                               bidirectional=self.bidirection,
                               bias=False)

        elif config.rnn == 'GRU':
            self.rnn = nn.GRU(input_size=config.embed_dim,
                              hidden_size=config.rnn_hidden_dim,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=self.bidirection,
                              bias=False)

        elif config.rnn == 'DAN':
            self.rnn = DAN(emb_dim=config.embed_dim,
                           hidden_dim=config.rnn_hidden_dim,)

        else:
            self.rnn = CNN(embedding_dim=self.D,
                           hidden_dim=config.rnn_hidden_dim,)

        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_embed).float(),
                                                      freeze=(not bool(config.train_word_embed)))  # V x D
        directions = 1
        if self.bidirection and config.rnn not in ['DAN','CNN']:
            directions = 2

        if self.config.marryup_type in 'input':
            self.linear = nn.Linear(directions * config.rnn_hidden_dim + self.RE_D, self.label_size)
            self.tag_embedding = nn.Parameter(torch.randn((self.label_size, self.RE_D)), requires_grad=True)
        elif self.config.marryup_type == 'output':
            self.linear = nn.Linear(directions * config.rnn_hidden_dim, self.label_size)
            self.weight = nn.Parameter(torch.randn(self.label_size))
        elif self.config.marryup_type == 'all':
            self.linear = nn.Linear(directions * config.rnn_hidden_dim + self.RE_D, self.label_size)
            self.weight = nn.Parameter(torch.randn(self.label_size))
            self.tag_embedding = nn.Parameter(torch.randn((self.label_size, self.RE_D)), requires_grad=True)
        elif self.config.marryup_type == 'none':
            self.linear = nn.Linear(directions * config.rnn_hidden_dim, self.label_size)

    def forward(self, input, lengths):
        # re_tags B x Label
        re_tags = self.fsa.forward(input, lengths)
        input = self.embedding(input)  # B x L x D
        if self.rnn in ['RNN', 'LSTM', 'GRU']:
            pack_padded_seq_input = torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True,
                                                                            enforce_sorted=False)
        else:
            pack_padded_seq_input = input

        B, L, D = input.size()

        if self.config.rnn in ['RNN', 'GRU']:
            out, hn = self.rnn(pack_padded_seq_input)  # B x L x H
        elif self.config.rnn == 'LSTM':
            out, (hn, cn) = self.rnn(pack_padded_seq_input)  # B x L x H
        else:
            hn = self.rnn(pack_padded_seq_input)

        if self.bidirection and (self.config.rnn in ['RNN', 'LSTM', 'GRU']):
            last_hidden = torch.cat((hn[0], hn[1]), 1)
        else:
            last_hidden = hn.squeeze()

        if self.config.marryup_type == 'input':
            margin_tensor = torch.tensor(0.5).cuda() if self.is_cuda else torch.tensor(0.5)
            tag = torch.matmul(re_tags, self.tag_embedding) # B x Lab, Lab x RE_D => B x RE_D
            tag_sum = torch.max(torch.sum(re_tags, dim=1), margin_tensor) # average
            tag = torch.div(tag.transpose(0,1), tag_sum).transpose(0,1)
            last_hidden_cat_re = torch.cat([last_hidden, tag], dim=1)
            score = self.linear(last_hidden_cat_re)
        elif self.config.marryup_type == 'output':
            score = self.linear(last_hidden)
            addlogits = torch.einsum('bl,l->bl', re_tags, self.weight)
            score = score + addlogits
        elif self.config.marryup_type == 'all':
            margin_tensor = torch.tensor(0.5).cuda() if self.is_cuda else torch.tensor(0.5)
            tag = torch.matmul(re_tags, self.tag_embedding) # B x Lab, Lab x RE_D => B x RE_D
            tag_sum = torch.max(torch.sum(re_tags, dim=1), margin_tensor) # average
            tag = torch.div(tag.transpose(0,1), tag_sum).transpose(0,1)
            last_hidden_cat_re = torch.cat([last_hidden, tag], dim=1)
            score = self.linear(last_hidden_cat_re)
            addlogits = torch.einsum('bl,l->bl', re_tags, self.weight)
            score = score + addlogits
        elif self.config.marryup_type == 'none':
            score = self.linear(last_hidden)

        return score