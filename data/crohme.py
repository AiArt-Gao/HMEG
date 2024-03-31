import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class CROHMELabelGraphDataset(Dataset):

    def __init__(self, root, names, nc, image_size=(256, 256)):
        super(CROHMELabelGraphDataset, self).__init__()
        self.img_size = image_size
        self.nc = nc + 1
        self.root = root
        self.npy_dir = os.path.join(root, 'link_npy')
        self.image_dir = os.path.join(root, 'Train_imgs')

        self.names = names
        self.npy_paths = [os.path.join(self.npy_dir, name + '.npy') for name in self.names]
        self.image_paths = [os.path.join(self.image_dir, name + '.png') for name in self.names]

        self.npy = [np.load(npy_path, allow_pickle=True).item() for npy_path in self.npy_paths]

        self.transform_64 = T.Compose([T.Resize((round(image_size[0] / 4), round(image_size[1] / 4))), T.ToTensor()])
        self.transform_128 = T.Compose([T.Resize((round(image_size[0] / 2), round(image_size[1] / 2))), T.ToTensor()])
        self.transform_256 = T.Compose([T.Resize(image_size), T.ToTensor()])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        img_path = self.image_paths[index]

        image = Image.open(img_path).convert('RGB')
        image_64 = self.transform_64(image)
        image_128 = self.transform_128(image)
        image_256 = self.transform_256(image)
        # image = torch.cat([image] * 3, dim=0)
        bbox = self.npy[index]['bbox']
        edge_type = self.npy[index]['edge_type']

        objs = bbox[:, 0].long()
        boxes = bbox[:, 1:]

        n = edge_type.shape[0]
        triples = []
        for row in range(n):
            for col in range(n):
                triples.append([row, edge_type[row, col], col])
        triples = torch.LongTensor(triples)
        # TODO layout gt
        layout = box2layout(boxes, img_size=(64, 64))
        return image_64, image_128, image_256, objs, boxes, layout, triples


# def box2layout(boxes, img_size=(256, 256)):
#     N = boxes.size(0)
#     x0, y0, x1, y1 = boxes.split(1, 1)
#     x0 = torch.round(x0 * img_size[1]).type(torch.long)
#     y0 = torch.round(y0 * img_size[0]).type(torch.long)
#     x1 = torch.round(x1 * img_size[1]).type(torch.long)
#     y1 = torch.round(y1 * img_size[0]).type(torch.long)
#     layout = torch.zeros(N, 1, *img_size, dtype=torch.float32)
#     for i in range(N):
#         layout[i, 0, y0[i]: y1[i], x0[i]: x1[i]] = 1
#     return layout

from model.layout import boxes_to_layout_matrix

def box2layout(boxes, img_size=(256, 256)):
    H, W = img_size
    return boxes_to_layout_matrix(boxes, H=H, W=W)


def crohme_collate_fn(batch):
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
    all_imgs_64, all_imgs_128, all_imgs_256, all_objs, all_boxes, all_layout, all_triples = [], [], [], [], [], [], []
    all_obj_to_img, all_triple_to_img = [], []
    obj_offset = 0
    for i, (img_64, img_128, img_256, objs, boxes, layout, triples) in enumerate(batch):
        all_imgs_64.append(img_64[None])
        all_imgs_128.append(img_128[None])
        all_imgs_256.append(img_256[None])
        O, T = objs.size(0), triples.size(0)
        all_objs.append(objs)
        all_boxes.append(boxes)
        all_layout.append(layout)
        triples = triples.clone()
        triples[:, 0] += obj_offset
        triples[:, 2] += obj_offset
        all_triples.append(triples)

        all_obj_to_img.append(torch.LongTensor(O).fill_(i))
        all_triple_to_img.append(torch.LongTensor(T).fill_(i))
        obj_offset += O

    all_imgs_64 = torch.cat(all_imgs_64)
    all_imgs_128 = torch.cat(all_imgs_128)
    all_imgs_256 = torch.cat(all_imgs_256)
    all_objs = torch.cat(all_objs)
    all_boxes = torch.cat(all_boxes)
    all_layout = torch.cat(all_layout)
    all_triples = torch.cat(all_triples)
    all_obj_to_img = torch.cat(all_obj_to_img)
    all_triple_to_img = torch.cat(all_triple_to_img)

    out = (all_imgs_64, all_imgs_128, all_imgs_256, all_objs, all_boxes, all_layout, all_triples,
           all_obj_to_img, all_triple_to_img)
    return out


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    root_dir = '../../datasets/crohme2019'
    npy_dir = os.path.join(root_dir, 'link_npy')
    names = [name[:-4] for name in os.listdir(npy_dir)]
    ds = CROHMELabelGraphDataset(root_dir, names, nc=102)

    loader_kwargs = {
        'batch_size': 8,
        'num_workers': 4,
        'shuffle': True,
        'collate_fn': crohme_collate_fn,
    }
    train_loader = DataLoader(ds, **loader_kwargs)
    ds_iter = iter(train_loader)
    data = next(ds_iter)
    image = data[2]
    objs = data[3]
    boxes = data[4]
    layout = data[5]
    triples = data[6]
    objs_to_imgs = data[7]
    print(image.shape)
    print(objs.shape)
    print(boxes.shape)
    print(layout.shape)
    print(triples.shape)
    print(objs_to_imgs.shape)


