import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from data.process import CROHME2Graph


class LGDataset(Dataset):

    def __init__(self, vocab, data_dir):
        super(LGDataset, self).__init__()
        self.data_dir = data_dir
        self.converter = CROHME2Graph(vocab)

        self.npy_paths = [os.path.join(self.data_dir, name)
                         for name in os.listdir(self.data_dir) if name.endswith('.npy')]
        self.npy = [np.load(npy_path, allow_pickle=True) for npy_path in self.npy_paths]


    def __len__(self):
        return len(self.npy)

    def __getitem__(self, index):
        objs = self.npy[index][0]
        triples = self.npy[index][1]
        return objs, triples


def lg_collate_fn(batch):
    """
    Collate function to be used when wrapping a VgSceneGraphDataset in a
    DataLoader. Returns a tuple of the following:

    - imgs: FloatTensor of shape (N, C, H, W)
    - objs: LongTensor of shape (O,) giving categories for all objects
    - boxes: FloatTensor of shape (O, 4) giving boxes for all objects
    - triples: FloatTensor of shape (T, 3) giving all triples, where
      triples[t] = [i, p, j] means that [objs[i], p, objs[j]] is a triple
    - obj_to_img: LongTensor of shape (O,) mapping objects to images;
      obj_to_img[i] = n means that objs[i] belongs to imgs[n]
    - triple_to_img: LongTensor of shape (T,) mapping triples to images;
      triple_to_img[t] = n means that triples[t] belongs to imgs[n].
    """
    # batch is a list, and each element is (image, objs, boxes, triples)
    all_objs, all_triples = [], []
    all_obj_to_img, all_triple_to_img = [], []
    obj_offset = 0
    for i, (objs, triples) in enumerate(batch):
        O, T = objs.size(0), triples.size(0)
        all_objs.append(objs)
        triples = triples.clone()
        triples[:, 0] += obj_offset
        triples[:, 2] += obj_offset
        all_triples.append(triples)

        all_obj_to_img.append(torch.LongTensor(O).fill_(i))
        all_triple_to_img.append(torch.LongTensor(T).fill_(i))
        obj_offset += O

    all_objs = torch.cat(all_objs)
    all_triples = torch.cat(all_triples)
    all_obj_to_img = torch.cat(all_obj_to_img)
    all_triple_to_img = torch.cat(all_triple_to_img)

    out = (all_objs, all_triples, all_obj_to_img, all_triple_to_img)
    return out





if __name__ == '__main__':
    import json
    from torch.utils.data import DataLoader

    CROHME_DIR = os.path.expanduser('../../datasets/crohme2019')

    with open(os.path.join(CROHME_DIR, 'vocab.json')) as f:
        vocab = json.load(f)

    lgds = LGDataset(vocab, '../100k_symlg')
    loader_kwargs = {
        'batch_size': 8,
        'num_workers': 4,
        'shuffle': True,
        'collate_fn': lg_collate_fn,
    }
    train_loader = DataLoader(lgds, **loader_kwargs)


    data_iter = iter(train_loader)
    objs, triples, obj_to_img, triple_to_img = next(data_iter)
    print(objs.shape)
    print(triples.shape)
    print(obj_to_img)
    print(triple_to_img.shape)






