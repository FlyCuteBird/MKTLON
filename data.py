
"""Loading data"""

import torch
import torch.utils.data as data
import os
import numpy as np

class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split):
        # path
        loc = data_path + '/'

        # Captions
        self.captions = []  #
        with open(loc + '%s_bert_caps.txt' % data_split, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())

        # Image features
        self.images = np.load(loc + '%s_ims.npy' % data_split, mmap_mode='r')  # image feature (Batch_Size, 36, 2048)
        self.length = len(self.captions)   # text length + 2

        print('image shape', self.images.shape)
        print('text shape', len(self.captions))

        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index // self.im_div
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]

        # Convert caption (string) to word ids.
        caps = []
        caps.extend(caption.decode().split(','))
        caps = list(map(int, caps))

        k = image.shape[0]
        assert k == 36
        captions = torch.Tensor(caps)
        return image, captions, index, img_id

    def __len__(self):
        return self.length


def collate_fn(data):
    """
    Build mini-batch tensors from a list of (image, caption) tuples.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids

def get_precomp_loader(data_path, data_split, opt, batch_size=64,
                       shuffle=True, num_workers=2):

    dset = PrecompDataset(data_path, data_split)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader


def get_loaders(data_name, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    train_loader = get_precomp_loader(dpath, 'train', opt,
                                      batch_size, True, workers)
    val_loader = get_precomp_loader(dpath, 'dev', opt,
                                    batch_size, False, workers)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    test_loader = get_precomp_loader(dpath, split_name, opt,
                                     batch_size, False, workers)
    return test_loader


