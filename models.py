import numpy as np
import torch
import layers
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class A3C_LSTM_GA(torch.nn.Module):

    def __init__(self, args):
        super(A3C_LSTM_GA, self).__init__()

        # Image Processing
        self.conv1 = nn.Conv2d(3, 128, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2)

        # dimension of state vector passed to A3C
        self.d_state_vector = 256

        # Instruction Processing
        self.use_new_fusion = args.use_new_fusion
        self.use_lang_enc = args.use_lang_enc
        self.word_embedding_dim = 32
        self.input_size = args.input_size
        self.embedding = nn.Embedding(self.input_size, self.word_embedding_dim)
        self.gated_attn_size = None

        if not self.use_new_fusion:
            if self.use_lang_enc:
                self.gated_attn_size = self.word_embedding_dim
                self.lang_encoder = layers.EncoderLayer(d_embed=self.word_embedding_dim, n_heads=4,
                        d_ff_hidden=20, dropout={'attn-self': 0.1, 'ff': 0.1})
            else:
                self.gru_hidden_size = self.d_state_vector
                self.gated_attn_size = self.gru_hidden_size
                self.gru = nn.GRUCell(self.word_embedding_dim, self.gru_hidden_size)

            # Gated-Attention layers
            self.attn_linear = nn.Linear(self.gated_attn_size, 64)

            self.linear = nn.Linear(64 * 8 * 17, 256)
        else:
            self.d_cnn_feat_map = 8 * 17
            self.n_cnn_feat_maps = 64
            self.attn = layers.MultiHeadAttentionLayer(n_heads=4,
                    d_src=self.d_cnn_feat_map, d_tgt=self.word_embedding_dim, dropout=0.1)

            self.d_gru_hidden = self.d_state_vector
            self.gru = nn.GRUCell(self.d_cnn_feat_map, self.d_gru_hidden)

            # @TODO might need full connected before recurrent reduce
            #       dim of feat maps
            self.linear = nn.Linear(self.d_gru_hidden, self.d_state_vector)

        # Time embedding layer, helps in stabilizing value prediction
        self.time_emb_dim = 32
        self.time_emb_layer = nn.Embedding(
                args.max_episode_length+1,
                self.time_emb_dim)

        # A3C-LSTM layers
        self.lstm = nn.LSTMCell(256, 256)
        self.critic_linear = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear = nn.Linear(256 + self.time_emb_dim, 3)

        # Initializing weights
        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train()

    def forward(self, inputs):
        x, input_inst, (tx, hx, cx) = inputs

        # Get the image representation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x_image_rep = F.relu(self.conv3(x))

        # Get the instruction representation
        if not self.use_new_fusion:
            if self.use_lang_enc:
                # extract word embeddings
                embedding_seq = self.embedding(input_inst[0, :]).unsqueeze(0).transpose(1,2)

                # encode each word with self attention, follow by a feed forward
                encoded_seq = self.lang_encoder(embedding_seq)

                # average across the sequence dimension to get the final instruction representation
                x_instr_rep = torch.mean(encoded_seq, dim=2)
            else:
                encoder_hidden = torch.zeros(1, self.gru_hidden_size)  # seq_len=1
                for i in range(input_inst.data.size(1)):
                    word_embedding = self.embedding(input_inst[0, i]).unsqueeze(0)
                    #print(word_embedding.shape)  # [1, 32]
                    encoder_hidden = self.gru(word_embedding, encoder_hidden)
                x_instr_rep = encoder_hidden.view(-1, encoder_hidden.size(1))

            # Get the attention vector from the instruction representation
            x_attention = torch.sigmoid(self.attn_linear(x_instr_rep))

            # Gated-Attention
            x_attention = x_attention.unsqueeze(2).unsqueeze(3)
            x_attention = x_attention.expand(1, 64, 8, 17)
            assert x_image_rep.size() == x_attention.size()
            x = x_image_rep*x_attention
            x = x.view(x.size(0), -1)
        else:
            # extract word embeddings
            instr_embedding_seq = self.embedding(input_inst[0, :]).unsqueeze(0).transpose(1,2)

            # collapse 2D feature maps in vectors
            x_image_rep = x_image_rep.view(x_image_rep.size(0), self.n_cnn_feat_maps, -1)

            # fuse vision feat maps with nat lang instructions with attention module
            x = self.attn(instr_embedding_seq.transpose(1,2), x_image_rep, x_image_rep)

            # combine feature maps corresponding to each instruction word using the gru
            gru_hidden = torch.zeros(1, self.d_gru_hidden)

            for i in range(x.size(2)):
                gru_hidden = self.gru(x[:,:,i], gru_hidden)
            x = gru_hidden

        # A3C-LSTM
        x = F.relu(self.linear(x))
        hx, cx = self.lstm(x, (hx, cx))
        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)

class A3C_LSTM_GA_OG(torch.nn.Module):

    def __init__(self, args):
        super(A3C_LSTM_GA_OG, self).__init__()

        # Image Processing
        self.conv1 = nn.Conv2d(3, 128, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2)

        # Instruction Processing
        self.gru_hidden_size = 256
        self.input_size = args.input_size
        self.embedding = nn.Embedding(self.input_size, 32)
        self.gru = nn.GRUCell(32, self.gru_hidden_size)

        # Gated-Attention layers
        self.attn_linear = nn.Linear(self.gru_hidden_size, 64)

        # Time embedding layer, helps in stabilizing value prediction
        self.time_emb_dim = 32
        self.time_emb_layer = nn.Embedding(
                args.max_episode_length+1,
                self.time_emb_dim)

        # A3C-LSTM layers
        self.linear = nn.Linear(64 * 8 * 17, 256)
        self.lstm = nn.LSTMCell(256, 256)
        self.critic_linear = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear = nn.Linear(256 + self.time_emb_dim, 3)

        # Initializing weights
        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train()

    def forward(self, inputs):
        x, input_inst, (tx, hx, cx) = inputs

        # Get the image representation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x_image_rep = F.relu(self.conv3(x))

        # Get the instruction representation
        encoder_hidden = torch.zeros(1, self.gru_hidden_size)  # seq_len=1
        for i in range(input_inst.data.size(1)):
            word_embedding = self.embedding(input_inst[0, i]).unsqueeze(0)
            #print(word_embedding.shape)  # [1, 32]
            encoder_hidden = self.gru(word_embedding, encoder_hidden)
        x_instr_rep = encoder_hidden.view(-1, encoder_hidden.size(1))
        #print(x_instr_rep.shape)
        # Get the attention vector from the instruction representation
        x_attention = torch.sigmoid(self.attn_linear(x_instr_rep))

        # Gated-Attention
        x_attention = x_attention.unsqueeze(2).unsqueeze(3)
        x_attention = x_attention.expand(1, 64, 8, 17)
        assert x_image_rep.size() == x_attention.size()
        x = x_image_rep*x_attention
        x = x.view(x.size(0), -1)

        # A3C-LSTM
        x = F.relu(self.linear(x))
        hx, cx = self.lstm(x, (hx, cx))
        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
