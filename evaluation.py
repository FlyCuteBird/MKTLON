"""Evaluation"""

from __future__ import print_function
import sys
from data import get_test_loader
from data import PrecompDataset
import time
import numpy as np
import torch
from model import CIBN
from collections import OrderedDict
import time
from torch.autograd import Variable
import logging
import tensorboard_logger as tb_logger


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):

        self.meters = OrderedDict()

    def update(self, k, v, n=0):

        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print):

    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None
    cap_lens = None


    max_n_word = 0
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))


    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        # tripe encoding --Q
        img_emb, cap_emb, cap_len = model.forward_emb(
            images, captions, lengths, volatile=True)

        if img_embs is None:
            if img_emb.dim() == 4:
                img_embs = np.zeros(
                    (len(data_loader.dataset), img_emb.size(1), img_emb.size(2), img_emb.size(3)))
            else:
                img_embs = np.zeros(
                    (len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
            cap_embs = np.zeros(
                (len(data_loader.dataset), cap_emb.size(1), max_n_word, cap_emb.size(3)))
            cap_lens = [0] * len(data_loader.dataset)


        # cache embeddings
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids, :, :max(lengths), :] = cap_emb.data.cpu().numpy().copy()

        for j, nid in enumerate(ids):
            cap_lens[nid] = cap_len[j]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time,
                        e_log=str(model.logger)))
        del images, captions
    return img_embs, cap_embs, cap_lens


def evalrank(model_path, data_path=None, split='dev', fold5=False):

    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    #print(opt)

    if data_path is not None:
        opt.data_path = data_path

    # construct model
    model = CIBN(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name,
                                  opt.batch_size, opt.workers, opt)

    print('Computing results...')
    img_embs, cap_embs, cap_lens = encode_data(
        model, data_loader)
    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] // 5, cap_embs.shape[0]))

    path = opt.model_name + '/' + opt.Matching_direction + '/'

    if not fold5:
        # no cross-validation, full evaluation
        img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
        start = time.time()
        sims = shard_xattn(model, img_embs, cap_embs,
                           cap_lens, opt, shard_size=64)

        # save similarity matrix for the fusion of different model
        np.save(path + '%s_%s_sim.npy' % (opt.Matching_direction, opt.belt_alpha), sims)

        end = time.time()
        print("calculate similarity time:", end - start)

        r, rt = i2t(img_embs, sims, return_ranks=True)
        ri, rti = t2i(img_embs, sims, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            img_embs_shard = img_embs[i * 5000:(i + 1) * 5000:5]
            cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]
            cap_lens_shard = cap_lens[i * 5000:(i + 1) * 5000]

            start = time.time()
            sims = shard_xattn(model, img_embs_shard, cap_embs_shard, cap_lens_shard, opt, shard_size=64)

            np.save(path + '%s_%s_sim.npy'% (opt.Matching_direction, i), sims)
            end = time.time()
            print("calculate similarity time:", end - start)

            r, rt0 = i2t(img_embs_shard, sims, return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(img_embs_shard, sims, return_ranks=True)
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[12]))
        print("Average i2t Recall: %.1f" % mean_metrics[10])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[11])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])

    torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')


def shard_xattn(model, images, captions, caplens, opt, shard_size=64):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    n_im_shard = (len(images) - 1) // shard_size + 1
    n_cap_shard = (len(captions) - 1) // shard_size + 1

    d = np.zeros((len(images), len(captions)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size * \
            i, min(shard_size * (i + 1), len(images))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_xattn batch (%d,%d)' % (i, j))
            cap_start, cap_end = shard_size * \
                j, min(shard_size * (j + 1), len(captions))
            with torch.no_grad():
                im = Variable(torch.from_numpy(
                    images[im_start:im_end])).cuda().float()
                s = Variable(torch.from_numpy(
                    captions[cap_start:cap_end])).cuda().float()
            # im = Variable(torch.from_numpy(
            #     images[im_start:im_end]), volatile=True).cuda().float()
            # s = Variable(torch.from_numpy(
            #     captions[cap_start:cap_end]), volatile=True).cuda().float()
            l = caplens[cap_start:cap_end]

            sim = model.forward_sim(im, s, l)
            d[im_start:im_end, cap_start:cap_end] = sim[0].data.cpu().numpy()+sim[1].data.cpu().numpy()+sim[2].data.cpu().numpy()
    sys.stdout.write('\n')
    return d


def i2t(images, sims, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = images.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, sims, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = images.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)
