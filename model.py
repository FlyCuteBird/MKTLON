""" CIBN model"""
import torch
import torch.nn as nn
import torch.nn.init
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from collections import OrderedDict
from sim_Reason import Reason_i2t, Reason_t2i
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from encoders import MemoryAugmentedEncoder
from attention import ScaledDotProductAttentionMemory, ScaledDotProductAttention


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

class WordEmbeddings(nn.Module):
    def __init__(self):
        super(WordEmbeddings, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
    def forward(self, x):
        with torch.no_grad():
            x = self.model(x)[0]
        x = x[-4:]
        x = torch.stack(x, dim=1)
        x = torch.sum(x, dim=1)
        return x

# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # bert embedding
        self.embedd = WordEmbeddings()

        # # one-hot embedding
        # self.embedd = nn.Embedding(30522,768)


        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(word_dim, embed_size, num_layers,
                          batch_first=True, bidirectional=use_bi_gru)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # bert_embedding
        x = self.embedd(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bi_gru:
            cap_emb = (cap_emb[:, :, :cap_emb.size(2) // 2] +
                       cap_emb[:, :, cap_emb.size(2) // 2:]) / 2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb, cap_len

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, opt, margin=0):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.Rank_Loss = opt.Rank_Loss

    def forward(self, scores):
        #self.opt.count_num = self.opt.count_num + 1
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        if self.opt.count_num == True:
            scores_dig = torch.diag(scores)
            for i in range(scores_dig.size(0)):
                pos = scores_dig[i].item()
                if torch.max(scores[i], 0)[0].item() == pos:
                    self.opt.positive_example.append(pos)
                    #neg = (torch.sum(scores[i]).item() - pos)/(scores[i].size(0)-1) - pos
                    ##### #neg = torch.mean(scores[i]).item() - pos
                    #####
                    # minScore = torch.min(scores[i], 0)[0].item()
                    # secMax, index = torch.topk(scores[i], 2)
                    # neg = (secMax[-1].item() + minScore)/2.0
                    #####
                    neg = torch.mean(scores[i] - pos).item()/(scores[i].size(0)-1)
                    #####
                    self.opt.negative_example.append(neg + self.opt.margin)
            pos_count = len(self.opt.positive_example)
            if pos_count > 100:
                pos = torch.tensor(self.opt.positive_example)
                neg = torch.tensor(self.opt.negative_example)
                mean_pos = pos.mean().cuda()
                mean_neg = neg.mean().cuda()
                stnd_pos = pos.std()
                stnd_neg = neg.std()

                A = stnd_pos.pow(2) - stnd_neg.pow(2)
                B = 2 * ((mean_pos * stnd_neg.pow(2)) - (mean_neg * stnd_pos.pow(2)))
                C = (mean_neg * stnd_pos).pow(2) - (mean_pos * stnd_neg).pow(2) + 2 * (stnd_pos * stnd_neg).pow(
                    2) * torch.log(self.opt.belt_alpha * stnd_neg / (stnd_pos) + 1e-8)

                thres = self.opt.margin
                self.opt.stnd_pos = stnd_pos.item()
                self.opt.stnd_neg = stnd_neg.item()
                self.opt.mean_pos = mean_pos.item()
                self.opt.mean_neg = mean_neg.item()

                E = B.pow(2) - 4 * A * C
                if E > 0:
                    up_margin = ((-B + torch.sqrt(E)) / (2 * A + 1e-10)).item()
                    if up_margin < 0.0:
                        up_margin = 0.0
                        self.opt.margin = thres
                    elif up_margin > 1.0:
                        up_margin = 0.2
                        self.opt.margin = 0.1 * up_margin + 0.9 * thres
                    else:
                        self.opt.margin = 0.1 * up_margin + 0.9 * thres

                    file_result = open(self.opt.result_name + '/' + '%s_%s_%s_margin.txt' % (
                        self.opt.data_name, self.opt.Matching_direction, self.opt.belt_alpha), 'a')
                    file_result.write(str(up_margin) + "\t" + str(self.opt.margin) + "\n")
                    file_result.close()

                else:
                    self.opt.margin = thres

                # caption retrieval
                cost_s = (self.opt.margin + scores - d1).clamp(min=0)
                # compare every diagonal score to scores in its row
                # image retrieval
                cost_im = (self.opt.margin + scores - d2).clamp(min=0)

                self.opt.negative_example = []
                self.opt.positive_example = []
                #self.opt.count_num = 0
            else:

                # caption retrieval
                cost_s = (self.opt.margin + scores - d1).clamp(min=0)
                # compare every diagonal score to scores in its row
                # image retrieval
                cost_im = (self.opt.margin + scores - d2).clamp(min=0)
        else:
            # compare every diagonal score to scores in its column
            # caption retrieval
            cost_s = (self.opt.margin + scores - d1).clamp(min=0)
            # compare every diagonal score to scores in its row
            # image retrieval
            cost_im = (self.opt.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the DynamciTopK, maximum or all violating negative for each query

        if self.Rank_Loss == 'DynamicTopK_Negative':
            topK = int((cost_s > 0.).sum() / (cost_s.size(0) + 0.00001) + 1)
            cost_s, index1 = torch.sort(cost_s, descending=True, dim=-1)
            cost_im, index2 = torch.sort(cost_im, descending=True, dim=0)

            return cost_s[:, 0:topK].sum() + cost_im[0:topK, :].sum()

        elif self.Rank_Loss == 'Hardest_Negative':
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

            return cost_s.sum() + cost_im.sum()

        else:
            return cost_s.sum() + cost_im.sum()


class CIBN(object):

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        # encoding multi_level visual information  # Choose one
        #self.UnifiedEncoding = MemoryAugmentedEncoder(opt.N, 0, attention_module=ScaledDotProductAttention, attention_module_kwargs={})
        self.UnifiedEncoding = MemoryAugmentedEncoder(opt.N, 0, attention_module=ScaledDotProductAttentionMemory, attention_module_kwargs={'m': 100})
        self.txt_enc = EncoderText(opt.word_dim, 2048, opt.num_layers, use_bi_gru=opt.bi_gru, no_txtnorm=opt.no_txtnorm)
        # Matching
        self.i2t_match = Reason_i2t(opt.feat_dim, opt.hid_dim, opt.out_dim, opt)
        self.t2i_match = Reason_t2i(opt.feat_dim, opt.hid_dim, opt.out_dim, opt)
        if torch.cuda.is_available():
            self.UnifiedEncoding.cuda()
            self.txt_enc.cuda()
            self.i2t_match.cuda()
            self.t2i_match.cuda()
            cudnn.benchmark = True
        # Loss and Optimizer
        self.criterion = ContrastiveLoss(opt=opt, margin=opt.margin)
        params = list(self.UnifiedEncoding.parameters())
        params += list(self.txt_enc.parameters())
        params += list(self.i2t_match.parameters())
        params += list(self.t2i_match.parameters())
        self.params = params
        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        self.Eiters = 0
        self.opt = opt

    def state_dict(self):
        state_dict = [self.UnifiedEncoding.state_dict(),
                      self.txt_enc.state_dict(),
                      self.i2t_match.state_dict(),
                      self.t2i_match.state_dict(),
                      ]
        return state_dict

    def load_state_dict(self, state_dict):

        self.UnifiedEncoding.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        self.i2t_match.load_state_dict(state_dict[2])
        self.t2i_match.load_state_dict(state_dict[3])

    def train_start(self):
        """switch to train mode
        """
        self.UnifiedEncoding.train()
        self.txt_enc.train()
        self.i2t_match.train()
        self.t2i_match.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.UnifiedEncoding.eval()
        self.txt_enc.eval()
        self.i2t_match.eval()
        self.t2i_match.eval()

    def forward_emb(self, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images, volatile=volatile)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # init encoding captions:
        cap_emb, cap_lens = self.txt_enc(captions, lengths)

        # shared encoding layers
        # img_emb-->[batch_size, N, n_regions, dim]
        # cap_emb-->[batch_size, N, n_regions, dim]
        img_emb = self.UnifiedEncoding(images)[0]
        cap_emb = self.UnifiedEncoding(cap_emb)[0]

        return img_emb, cap_emb, cap_lens

    def forward_sim(self, img_emb, cap_emb, cap_lens):
        if self.opt.Matching_direction == 'i2t':
            i2t_scores = self.i2t_match(img_emb, cap_emb, cap_lens, self.opt)
            return i2t_scores
        elif self.opt.Matching_direction == 't2i':
            t2i_scores = self.t2i_match(img_emb, cap_emb, cap_lens, self.opt)
            return t2i_scores
        else:
            t2i_scores = self.t2i_match(img_emb, cap_emb, cap_lens, self.opt)
            i2t_scores = self.i2t_match(img_emb, cap_emb, cap_lens, self.opt)
            return t2i_scores + i2t_scores

    def forward_loss(self, scores, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(scores)
        self.logger.update('Le', loss.item())
        return loss

    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        img_emb, cap_emb, cap_lens = self.forward_emb(
            images, captions, lengths)


        scores = self.forward_sim(img_emb, cap_emb, cap_lens)
        scores = scores[0]+scores[1]+scores[2]

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(scores)

        # # log Loss
        # file = open(self.opt.model_name + '/' + self.opt.region_relation + '/' + '%s_%s_Loss.txt' %(self.opt.region_relation, self.opt.windows_size), 'a' )
        # file.write(str(self.Eiters) + "    " + str(loss.item()) + "\n")
        # file.close()

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
