from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from mean_out import MeanOut
import torch
import torch.nn as nn
import numpy as np
from graph_Convolution import VisualConvolution, TextualConvolution
from collections import OrderedDict



def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def inter_relations(K, Q, xlambda):
    """
    Q: (batch, queryL, d)
    K: (batch, sourceL, d)
    return (batch, queryL, sourceL)
    """
    batch_size, queryL = Q.size(0), Q.size(1)
    batch_size, sourceL = K.size(0), K.size(1)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    queryT = torch.transpose(Q, 1, 2)

    attn = torch.bmm(K, queryT)

    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax(dim=1)(attn * xlambda)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)

    return attn

class WeightNetwork(nn.Module):
    def __init__(self, input_dim):
        super(WeightNetwork, self).__init__()
        self.input_dim = input_dim
        self.W1 = nn.Linear(input_dim, input_dim)
        self.W2 = nn.Linear(input_dim, input_dim)
        self.init_para()

    def init_para(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
    def forward(self, features):
        W1_features = self.W1(features)
        W2_features = self.W2(features)
        weights = torch.softmax(torch.bmm(W1_features, W2_features.permute(0, 2, 1)), dim=-1)
        return weights

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param
        super(WeightNetwork, self).load_state_dict(new_state)

class Encoding_sim(nn.Module):
    def __init__(self, embed_size):
        super(Encoding_sim, self).__init__()
        self.embed_dim = embed_size

    def forward(self, nodes1, nodes2, num_block, xlambda):
        batch_size, num_element = nodes2.size(0), nodes2.size(1)
        # local matching
        inter_relation = inter_relations(nodes1, nodes2, xlambda)
        attnT = torch.transpose(inter_relation, 1, 2)
        featureT = torch.transpose(nodes1, 1, 2)
        weightFeature = torch.bmm(featureT, attnT)
        weightFeatureT = torch.transpose(weightFeature, 1, 2)
        qry_set = nodes2.view(batch_size, num_element, num_block, -1)
        ctx_set = weightFeatureT.view(batch_size, num_element, num_block, -1)
        # (batch_size, num_element, num_block)
        sim_mvector = cosine_similarity(qry_set, ctx_set, dim=-1)
        return sim_mvector

    def load_state_dict(self, state_dict):

        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param
        super(Encoding_sim, self).load_state_dict(new_state)


class Reason_i2t(nn.Module):

    def __init__(self,
                 feat_dim,
                 hid_dim,
                 out_dim,
                 opt):
        '''
        ## Variables:
        - feat_dim: dimensionality of input image features
        - out_dim: dimensionality of the output
        - dropout: dropout probability
        '''

        super(Reason_i2t, self).__init__()

        # Set parameters
        self.feat_dim = feat_dim
        self.out_dim = out_dim
        self.opt = opt

        # para
        self.delt = nn.Parameter(torch.FloatTensor(opt.num_block))

        self.Encoding_sim = Encoding_sim(opt.embed_size)

        # reasoning mean
        self.mean_out = MeanOut(opt.num_block, hid_dim, opt.Pieces)

        # reasoning matching scores
        self.graph_weights = WeightNetwork(opt.num_block)
        self.graph_reasoning = VisualConvolution(opt.num_block, opt.num_block, n_kernels=8, bias=True)

        # self.out_1 = nn.utils.weight_norm(nn.Linear(opt.num_block, hid_dim))
        self.out_2 = nn.utils.weight_norm(nn.Linear(hid_dim, out_dim))

        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.delt, 0.5, 1)
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)

    def load_state_dict(self, state_dict):

        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param
        super(Reason_i2t, self).load_state_dict(new_state)

    def i2t_scores(self, tnodes, vnodes, num_block):
        batch_size, num_regions = vnodes.size(0), vnodes.size(1)
        # 9.0 for Flickr30K dataset; 10.0 for MSCOCO dataset
        sim_mvector = self.Encoding_sim(tnodes, vnodes, num_block, 9.0)
        sim_mvector = sim_mvector * self.delt
        # graph reasoning
        graph_weights = self.graph_weights(sim_mvector)
        sim_mvector = self.graph_reasoning(sim_mvector, graph_weights)
        
        # sim = self.out_2(self.mean_out(sim_mvector.view(batch_size * num_regions, -1)).view(batch_size, num_regions, -1).tanh())
        # sim = sim.view(batch_size, -1).mean(dim=1, keepdim=True)
        sim = self.mean_out(sim_mvector.view(batch_size * num_regions, -1)).view(batch_size, num_regions, -1)
        sim = self.out_2(sim).tanh()
        sim = sim.view(batch_size, -1).mean(dim=1, keepdim=True)

        return sim

    def forward(self, images, captions, cap_lens, opt):
        similarities = []
        similarities_list = []
        n_image, n_caption = images.size(0), captions.size(0)
        for j in range(opt.N):
            for i in range(n_caption):
                # Get the i-th text description
                n_word = cap_lens[i]
                cap_i = captions[:,j,:,:][i, :n_word, :].unsqueeze(0).contiguous()
                # --> (n_image, n_word, d)
                cap_i_expand = cap_i.repeat(n_image, 1, 1)

                # --> compute similarity between query region and context word
                i2t = self.i2t_scores(
                    cap_i_expand, images[:,j,:,:], opt.num_block)
                similarities.append(i2t)
            similarities = torch.cat(similarities, 1)
            similarities_list.append(similarities)
            similarities = []

        return similarities_list


class Reason_t2i(nn.Module):

    def __init__(self,
                 feat_dim,
                 hid_dim,
                 out_dim,
                 opt):

        super(Reason_t2i, self).__init__()

        # Set parameters
        self.feat_dim = feat_dim
        self.out_dim = out_dim
        self.opt = opt
        # t2i feature transfer
        self.t2i_transfer = nn.Linear(opt.embed_size, opt.embed_size)

        self.Encoding_sim = Encoding_sim(opt.embed_size)

        self.graph_weights = WeightNetwork(opt.num_block)
        self.graph_reasoning = VisualConvolution(opt.num_block, opt.num_block, n_kernels=4, bias=True)
        self.sigmoid = nn.Sigmoid()

        # reasoning matching scores
        # self.out_1 = nn.utils.weight_norm(nn.Linear(opt.num_block, hid_dim))
        # reasoning mean
        self.mean_out = MeanOut(opt.num_block, hid_dim, opt.Pieces)
        self.out_2 = nn.utils.weight_norm(nn.Linear(hid_dim, out_dim))

        # para
        self.delt = nn.Parameter(torch.FloatTensor(opt.num_block))
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.delt, 1, 1)
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)

    def load_state_dict(self, state_dict):

        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param
        super(Reason_t2i, self).load_state_dict(new_state)

    def t2i_scores(self, vnodes, tnodes, num_block):
        batch_size, num_words = tnodes.size(0), tnodes.size(1)
        sim_mvector = self.Encoding_sim(vnodes, tnodes, num_block, 20.0)
        sim_mvector = sim_mvector * self.delt
        graph_weights = self.graph_weights(sim_mvector)
        sim_mvector = self.graph_reasoning(sim_mvector, graph_weights)

        # sim = self.out_2(self.mean_out(sim_mvector.view(batch_size * num_words, -1)).view(batch_size, num_words, -1).tanh())
        # sim = sim.view(batch_size, -1).mean(dim=1, keepdim=True)

        sim = self.mean_out(sim_mvector.view(batch_size * num_words, -1)).view(batch_size, num_words, -1)
        sim = self.out_2(sim).tanh()
        sim = sim.view(batch_size, -1).mean(dim=1, keepdim=True)
        return sim

    def forward(self, images, captions, cap_lens, opt):
       # Multi-layers Encoding
       # images--->[batch_size, N_layers, n_regions, embed_dim]
       # captions->[batch_size, N_layers, n_regions, embed_dim]
        n_image = images.size(0)
        n_caption = captions.size(0)
        similarities = []
        similarities_list = []
        for j in range(opt.N):
            for i in range(n_caption):
                # Get the i-th text descriptiontanh()
                n_word = cap_lens[i]
                cap_i = captions[:,j,:,:][i, :n_word, :].unsqueeze(0).contiguous()
                # --> (n_image, n_word, d)
                cap_i_expand = cap_i.repeat(n_image, 1, 1)
                t2i = self.t2i_scores(
                    images[:,j,:,:], cap_i_expand, opt.num_block)
                similarities.append(t2i)

            # (n_image, n_caption)
            similarities = torch.cat(similarities, 1)
            similarities_list.append(similarities)

            similarities = []

        return similarities_list
