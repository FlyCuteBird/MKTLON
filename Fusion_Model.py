'''
The code is used to merge different results to obtain a new result.
'''

import numpy as np


def i2t(im_len, sims, npts=None, return_ranks=False):
    npts = im_len
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


def t2i(im_len, sims, npts=None, return_ranks=False):
    npts = im_len
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


if __name__ == '__main__':
    # reading  the similarity matrix
    # coco or flickr30k
    dataset_name = 'flickr30k'
    isfold5 = False
    mid_result = []
    results = []
    if dataset_name == 'coco' and isfold5 == True:
        for i in range(5):
            print(i)
            sim1 = np.load('./0.1/t2i_{}_sim.npy'.format(i))
            sim2 = np.load('./0.1/i2t_{}_sim.npy'.format(i))
            ima_len, caps_len = sim1.shape
            for j in range(101):
                alpha = j / 100.0
                sims = alpha * sim1 + (1-alpha) * sim2
                r, rt = i2t(ima_len, sims, return_ranks=True)
                ri, rti = t2i(ima_len, sims, return_ranks=True)
                ar = (r[0] + r[1] + r[2]) / 3
                ari = (ri[0] + ri[1] + ri[2]) / 3
                rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
                mid_result.append((rsum, r[0], r[1], r[2], ri[0], ri[1], ri[2], i))
            print(max(mid_result))
            results.append(list(max(mid_result)))
            mid_result = []
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print(len(mean_metrics))
        print(mean_metrics)
        print("rsum: %.1f" % (mean_metrics[0]))
        print("Image to text: %.1f %.1f %.1f" %
              mean_metrics[1:4])
        print("Text to image: %.1f %.1f %.1f" %
              mean_metrics[4:7])
    else:
        sim1 = np.load('./0.1/t2i_0.1_sim.npy')
        sim2 = np.load('./0.3/i2t_0.3_sim.npy')
        ima_len, caps_len = sim1.shape
        for i in range(101):
            alpha = i / 100.0
            print(alpha)
            sims = alpha * sim1 + (1 - alpha) * sim2
            r, rt = i2t(ima_len, sims, return_ranks=True)
            ri, rti = t2i(ima_len, sims, return_ranks=True)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            mid_result.append((rsum, r[0], r[1], r[2], ri[0], ri[1], ri[2], i))
        print(mid_result)
        print(max(mid_result))





