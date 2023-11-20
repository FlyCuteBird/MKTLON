""" Training """

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time
import shutil
import torch
import numpy
import Parser
import data
import random
import numpy as np
from model import CIBN
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, shard_xattn
import logging
import tensorboard_logger as tb_logger

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

random.seed(2022)
torch.manual_seed(2022)
np.random.seed(2022)

def main():

    opt = Parser.parse_opt()
    print(opt)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    # Load data loaders
    train_loader, val_loader = data.get_loaders(
        opt.data_name, opt.batch_size, opt.workers, opt)

    # Construct the model
    model = CIBN(opt)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            validate(opt, val_loader, model)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # Train the Model
    best_rsum = 0
    for epoch in range(opt.num_epochs):
        print(opt.logger_name)
        print(opt.model_name)

        adjust_learning_rate(opt, model.optimizer, epoch)

        # train for one epoch
        train(opt, train_loader, model, epoch, val_loader)

        # evaluate on validation set
        rsum = validate(opt, val_loader, model)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if not os.path.exists(opt.model_name + '/' + opt.region_relation + '/'):
            os.makedirs(opt.model_name + '/' + opt.region_relation + '/')

        if opt.region_relation == 'NTN_relation':
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.Eiters,
            }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch%5), prefix=opt.model_name + '/' + opt.Matching_direction + '/')

        else:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.Eiters,
            }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch%5),
                prefix=opt.model_name +'/' + opt.Matching_direction + '/')

def train(opt, train_loader, model, epoch, val_loader):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        # switch to train mode
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(*train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # validate at every val_step
        if model.Eiters % opt.val_step == 0:
            validate(opt, val_loader, model)

def validate(opt, val_loader, model):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs, cap_lens = encode_data(
        model, val_loader, opt.log_step, logging.info)
    img_embs = numpy.array([img_embs[i] for i in range(0, len(img_embs), 5)])
    start = time.time()
    sims = shard_xattn(model, img_embs, cap_embs, cap_lens,
                       opt, shard_size=64)  # changed from 128 to 64
    end = time.time()
    print("calculate similarity time:", end - start)

    # caption retrieval

    (r1, r5, r10, medr, meanr) = i2t(img_embs, sims)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(
        img_embs, sims)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanr))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    file_result = open(opt.result_name + '/' + '%s_%s_result.txt' %(opt.data_name, opt.Matching_direction),'a')
    file_result.write("i2t: " + str(r1) + "  " + str(r5) + "  " + str(r10) + "  " +
                      "t2i: " + str(r1i) + "  " + str(r5i) + "  " + str(r10i) + "  " +
                      "rsum: " + str(currscore) + "\n")
    file_result.close()

    return currscore


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                shutil.copyfile(prefix + filename, prefix +
                                'model_best.pth.tar')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
