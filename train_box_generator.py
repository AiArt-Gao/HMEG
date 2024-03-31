import os
import json
import random
import time
import numpy as np

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision

from data.crohme_box import CROHMELabelGraphDataset, crohme_collate_fn
from data.lg_dataset import LGDataset, lg_collate_fn
from data.utils import RecursiveIter, view_box
from model.bbox_net import BBoxNet, LayoutDiscriminator


CROHME_DIR = os.path.expanduser('../datasets/crohme2019')
LATEX2IMG_DIR = os.path.expanduser('../datasets/100k_npy')


class BBoxTrainer(object):

    def __init__(self, batch_size, loader_workers):
        self.device = 'cuda'
        self.save_dir = 'weight'
        os.makedirs(self.save_dir, exist_ok=True)
        self.noise_dim = 64

        self.batch_size = batch_size
        self.loader_workers = loader_workers

        self.vocab, self.train_loader = self._build_crohme_loaders()
        self.add_loader = self._build_100k_loaders(self.vocab)
        self.train_iter = RecursiveIter(self.train_loader)
        self.add_iter = RecursiveIter(self.add_loader)

        self.generator = BBoxNet(self.vocab).to(self.device)
        self.discriminator = LayoutDiscriminator(self.vocab).to(self.device)
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=5e-5)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=5e-5, weight_decay=1e-4)

        self.mse_loss_func = torch.nn.MSELoss()
        self.l1_loss_func = torch.nn.L1Loss()
        with torch.no_grad():
            self.valid = torch.FloatTensor(batch_size, 1).fill_(1.0).to(self.device)
            self.fake = torch.FloatTensor(batch_size, 1).fill_(0.0).to(self.device)
        self.writer = SummaryWriter()

    def _build_crohme_dsets(self):
        with open(os.path.join(CROHME_DIR, 'vocab.json')) as f:
            vocab = json.load(f)
        nc = len(vocab['object_idx_to_name'])
        npy_dir = os.path.join(CROHME_DIR, 'link_npy')
        names = [name[:-4] for name in os.listdir(npy_dir)]
        random.shuffle(names)
        train_dset = CROHMELabelGraphDataset(CROHME_DIR, names, nc=nc)
        return vocab, train_dset

    def _build_crohme_loaders(self):
        vocab, train_dset = self._build_crohme_dsets()
        collate_fn = crohme_collate_fn
        loader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': self.loader_workers,
            'shuffle': True,
            'collate_fn': collate_fn,
            'drop_last': True
        }
        train_loader = DataLoader(train_dset, **loader_kwargs)
        return vocab, train_loader

    def _build_100k_loaders(self, vocab):
        data_dset = LGDataset(vocab, LATEX2IMG_DIR)
        collate_fn = lg_collate_fn
        loader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': self.loader_workers,
            'shuffle': True,
            'collate_fn': collate_fn,
            'drop_last': True
        }
        data_loader = DataLoader(data_dset, **loader_kwargs)
        return data_loader

    def train(self, n_iter=100000000):
        start = time.time()
        for i_iter in range(n_iter):
            for p in self.discriminator.parameters():
                p.requires_grad_(True)
            self.optimizer_d.zero_grad()
            with torch.no_grad():
                objs, boxes, triples, obj_to_img, triple_to_img = next(self.train_iter)
                objs = objs.to(self.device)
                triples = triples.to(self.device)
                boxes = boxes.to(self.device)
                obj_to_img = obj_to_img.to(self.device)
                triple_to_img = triple_to_img.to(self.device)

            objs_sampled, boxes_sampled, triples_sampled, obj_to_img_sampled = \
                self.sample_batch_graph(objs, boxes, triples, obj_to_img, triple_to_img)
            real = self.discriminator(objs_sampled, boxes_sampled, triples_sampled, obj_to_img_sampled)
            with torch.no_grad():
                if np.random.randint(2):
                    objs, triples, obj_to_img, triple_to_img = next(self.add_iter)
                else:
                    objs, _, triples, obj_to_img, triple_to_img = next(self.train_iter)
                objs = objs.to(self.device)
                triples = triples.to(self.device)
                obj_to_img = obj_to_img.to(self.device)
                triple_to_img = triple_to_img.to(self.device)
            noise = gen_rand_noise(objs.size(0), self.noise_dim, self.device)
            pred_boxes = self.generator(objs, triples, noise, obj_to_img)

            objs_sampled, boxes_sampled, triples_sampled, obj_to_img_sampled = \
                self.sample_batch_graph(objs, pred_boxes, triples, obj_to_img, triple_to_img)
            fake = self.discriminator(objs_sampled, boxes_sampled, triples_sampled, obj_to_img_sampled)
            fake_loss = self.mse_loss_func(fake, self.fake)
            real_loss = self.mse_loss_func(real, self.valid)
            d_loss = (real_loss + fake_loss) / 2 * 0.2
            d_loss.backward()
            self.optimizer_d.step()

            for p in self.discriminator.parameters():
                p.requires_grad_(False)
            self.optimizer_g.zero_grad()
            with torch.no_grad():
                objs, boxes, triples, obj_to_img, triple_to_img = next(self.train_iter)
                objs = objs.to(self.device)
                boxes = boxes.to(self.device)
                triples = triples.to(self.device)
                obj_to_img = obj_to_img.to(self.device)
                triple_to_img = triple_to_img.to(self.device)
            noise = gen_rand_noise(objs.size(0), self.noise_dim, self.device)
            pred_boxes = self.generator(objs, triples, noise, obj_to_img)
            box_loss = self.l1_loss_func(pred_boxes, boxes)
            with torch.no_grad():
                if np.random.randint(2):
                    objs, triples, obj_to_img, triple_to_img = next(self.add_iter)
                else:
                    objs, _, triples, obj_to_img, triple_to_img = next(self.train_iter)
                objs = objs.to(self.device)
                triples = triples.to(self.device)
                obj_to_img = obj_to_img.to(self.device)
                triple_to_img = triple_to_img.to(self.device)
            noise = gen_rand_noise(objs.size(0), self.noise_dim, self.device)
            pred_boxes = self.generator(objs, triples, noise, obj_to_img)

            objs_sampled, boxes_sampled, triples_sampled, obj_to_img_sampled = \
                self.sample_batch_graph(objs, pred_boxes, triples, obj_to_img, triple_to_img)
            fake = self.discriminator(objs_sampled, boxes_sampled, triples_sampled, obj_to_img_sampled)
            dis_loss = self.mse_loss_func(fake, self.valid)
            g_loss = box_loss * 10 + dis_loss * 0.2
            g_loss.backward()
            self.optimizer_g.step()

            if i_iter % 10 == 0:
                cost = time.time() - start
                print('step %d  g_box_loss %.4f  g_fake_loss %.4f  d_fake_loss %.4f  d_real_loss %.4f  cost %.4f'
                      % (i_iter, box_loss, dis_loss, fake_loss, real_loss, cost))

                self.writer.add_scalars('gan', {'g_loss': dis_loss, 'd_fake': fake_loss, 'd_real': real_loss}, i_iter)
                self.writer.add_scalar('box_loss', box_loss, i_iter)
                start = time.time()
            if i_iter % 100 == 0:
                self.eval(i_iter)
            if i_iter % 10000 == 0:
                self.save(i_iter)

    def save(self, i_iter):
        checkpoint = {
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
        }
        model_path = os.path.join(self.save_dir, 'box_net_%d.pkl' % i_iter)
        torch.save(checkpoint, model_path)
        print('save model: %s' % model_path)

    def eval(self, i_iter):
        layouts = []
        with torch.no_grad():
            objs, boxes, triples, obj_to_img, _ = next(self.train_iter)
            objs = objs.to(self.device)
            triples = triples.to(self.device)
            obj_to_img = obj_to_img.to(self.device)
        noise = gen_rand_noise(objs.size(0), self.noise_dim, self.device)
        pred_boxes = self.generator(objs, triples, noise, obj_to_img).detach().cpu()
        layouts.append(view_box(pred_boxes, obj_to_img))
        with torch.no_grad():
            objs, triples, obj_to_img, _ = next(self.add_iter)
            objs = objs.to(self.device)
            triples = triples.to(self.device)
            obj_to_img = obj_to_img.to(self.device)
        noise = gen_rand_noise(objs.size(0), self.noise_dim, self.device)
        pred_boxes = self.generator(objs, triples, noise, obj_to_img).detach().cpu()
        layouts.append(view_box(pred_boxes, obj_to_img))
        layouts = torch.cat(layouts, dim=0)

        grid_images = torchvision.utils.make_grid(layouts, nrow=self.batch_size, padding=0)
        self.writer.add_image('eval_images', grid_images, i_iter)

    def sample_batch_graph(self, objs, boxes, triples, obj_to_img, triple_to_img):
        N = obj_to_img.data.max().item() + 1
        offset_org = 0
        offset_tar = 0
        objs_sampled = []
        boxes_sampled = []
        triples_sampled = []
        obj_to_img_sampled = []
        for n in range(N):
            box_idx = (obj_to_img.data == n).nonzero().view(-1)
            triple_idx = (triple_to_img.data == n).nonzero().view(-1)
            # print(triples.device)
            graph_objs, graph_boxes, graph_triples = self.sample_graph(objs[box_idx], boxes[box_idx],
                                                                  triples[triple_idx].clone(),
                                                                  offset=offset_org)
            # graph_boxes += offset_tar
            graph_triples[:, [0, 2]] += offset_tar
            objs_sampled.append(graph_objs)
            boxes_sampled.append(graph_boxes)
            triples_sampled.append(graph_triples)
            obj_to_img_sampled.append(torch.LongTensor(graph_boxes.size(0)).fill_(n))
            offset_org += box_idx.size(0)
            offset_tar += graph_boxes.size(0)
        objs_sampled = torch.cat(objs_sampled, dim=0)
        boxes_sampled = torch.cat(boxes_sampled, dim=0)
        triples_sampled = torch.cat(triples_sampled, dim=0)
        obj_to_img_sampled = torch.cat(obj_to_img_sampled, dim=0).to(obj_to_img.device)
        return objs_sampled, boxes_sampled, triples_sampled, obj_to_img_sampled

    def sample_graph(self, objs, boxes, triples, offset=0, max_op_edge=5):
        # print(triples.device)
        O = boxes.size(0)
        triples[:, [0, 2]] -= offset
        idx = torch.tensor(np.random.randint(O)).view(-1).to(triples.device)
        required_edge_idx = (((triples[:, 2] == idx) | (triples[:, 0] == idx)) & (triples[:, 1] != 0)).nonzero().view(
            -1)
        optional_edge_idx = ((triples[:, 2] == idx) & (triples[:, 1] == 0)).nonzero().view(-1)
        n_op = optional_edge_idx.size(0)
        optional_edge_idx = optional_edge_idx[np.random.choice(np.arange(n_op), min(max_op_edge, n_op), replace=False)]
        selected_triples = torch.cat([triples[required_edge_idx], triples[optional_edge_idx]], dim=0)
        # selected_triples = triples[required_edge_idx].view(-1, 3)
        boxes_idx = torch.cat([idx, selected_triples[:, 0], selected_triples[:, 2]]).unique(sorted=False)
        # print(boxes_idx.device)
        # print(selected_triples.device)
        selected_triples[:, 0] = find_index(boxes_idx, selected_triples[:, 0])
        selected_triples[:, 2] = find_index(boxes_idx, selected_triples[:, 2])
        selected_boxes = boxes[boxes_idx]
        selected_objs = objs[boxes_idx]
        return selected_objs, selected_boxes, selected_triples


def gen_rand_noise(n_node, dim, device):
    noise = torch.randn(n_node, dim)
    noise = noise.to(device)
    return noise


def find_index(a, b):
    c = a.expand(b.size(0), -1)
    d = b.view(-1, 1)
    pos = (c == d).nonzero()
    pos = pos[:, 1].view(-1)
    # print(b)
    # print(c)
    # print(pos)
    # print('-----------------')
    return pos


if __name__ == '__main__':
    torch.cuda.set_device(4)
    trainer = BBoxTrainer(64, 4)
    trainer.train()

    # objs = torch.tensor([1, 2, 3, 4, 1, 2, 3, 4])
    #
    # boxes = torch.tensor([
    #     [1] * 4,
    #     [2] * 4,
    #     [3] * 4,
    #     [4] * 4,
    #     [5] * 4,
    #     [6] * 4,
    #     [7] * 4,
    #     [8] * 4,
    # ])
    # triples = torch.tensor([
    #     [0, 0, 3],
    #     [0, 0, 2],
    #     [0, 0, 1],
    #     [1, 0, 0],
    #     [1, 0, 2],
    #     [1, 0, 3],
    #     [2, 0, 0],
    #     [2, 0, 1],
    #     [2, 0, 3],
    #     [3, 0, 0],
    #     [3, 0, 1],
    #     [3, 0, 2],
    #     [2, 1, 1],
    #     [1, 1, 2],
    #     [3, 1, 0],
    #     [0, 1, 3],
    #     [1, 1, 0],
    #     [4, 0, 7],
    #     [4, 0, 6],
    #     [4, 0, 5],
    #     [5, 0, 4],
    #     [5, 0, 6],
    #     [5, 0, 7],
    #     [6, 0, 4],
    #     [6, 0, 5],
    #     [6, 0, 7],
    #     [7, 0, 4],
    #     [7, 0, 5],
    #     [7, 0, 6],
    #     [6, 1, 5],
    #     [5, 1, 6],
    #     [7, 1, 4],
    #     [4, 1, 7],
    #     [5, 1, 4]
    # ])
    #
    # obj_to_img = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    # triple_to_img = torch.tensor([0] * 17 + [1] * 17)
    #
    #
    # def sample_graph(objs, boxes, triples, offset=0, max_op_edge=3):
    #     # print(triples.device)
    #     O = boxes.size(0)
    #     triples[:, [0, 2]] -= offset
    #     idx = torch.tensor(np.random.randint(O)).view(-1).to(triples.device)
    #     required_edge_idx = (((triples[:, 2] == idx) | (triples[:, 0] == idx)) & (triples[:, 1] != 0)).nonzero().view(
    #         -1)
    #     # optional_edge_idx = ((triples[:, 2] == idx) & (triples[:, 1] == 0)).nonzero().view(-1)
    #     # n_op = optional_edge_idx.size(0)
    #     # optional_edge_idx = optional_edge_idx[np.random.choice(np.arange(n_op), min(max_op_edge, n_op), replace=False)]
    #     # selected_triples = torch.cat([triples[required_edge_idx], triples[optional_edge_idx]], dim=0)
    #     selected_triples = triples[required_edge_idx].view(-1, 3)
    #     boxes_idx = torch.cat([idx, selected_triples[:, 0], selected_triples[:, 2]]).unique(sorted=False)
    #     # print(boxes_idx.device)
    #     # print(selected_triples.device)
    #     selected_triples[:, 0] = find_index(boxes_idx, selected_triples[:, 0])
    #     selected_triples[:, 2] = find_index(boxes_idx, selected_triples[:, 2])
    #     selected_boxes = boxes[boxes_idx]
    #     selected_objs = objs[boxes_idx]
    #     return selected_objs, selected_boxes, selected_triples
    #
    # def sample_batch_graph(objs, boxes, triples, obj_to_img, triple_to_img):
    #     N = obj_to_img.data.max().item() + 1
    #     offset_org = 0
    #     offset_tar = 0
    #     objs_sampled = []
    #     boxes_sampled = []
    #     triples_sampled = []
    #     obj_to_img_sampled = []
    #     for n in range(N):
    #         box_idx = (obj_to_img.data == n).nonzero().view(-1)
    #         triple_idx = (triple_to_img.data == n).nonzero().view(-1)
    #         graph_objs, graph_boxes, graph_triples = sample_graph(objs[box_idx], boxes[box_idx],
    #                                                               triples[triple_idx].clone(),
    #                                                               offset=offset_org)
    #         # graph_boxes += offset_tar
    #         graph_triples[:, [0, 2]] += offset_tar
    #         objs_sampled.append(graph_objs)
    #         boxes_sampled.append(graph_boxes)
    #         triples_sampled.append(graph_triples)
    #         obj_to_img_sampled.append(torch.LongTensor(graph_boxes.size(0)).fill_(n))
    #         offset_org += box_idx.size(0)
    #         offset_tar += graph_boxes.size(0)
    #     objs_sampled = torch.cat(objs_sampled, dim=0)
    #     boxes_sampled = torch.cat(boxes_sampled, dim=0)
    #     triples_sampled = torch.cat(triples_sampled, dim=0)
    #     obj_to_img_sampled = torch.cat(obj_to_img_sampled, dim=0)
    #     return objs_sampled, boxes_sampled, triples_sampled, obj_to_img_sampled
    #
    #
    # s_objs, s_boxes, s_triples, s_obj_to_img = sample_batch_graph(objs, boxes, triples, obj_to_img, triple_to_img)
    # print('--------------------')
    # print(s_objs)
    # print(s_boxes)
    # print(s_triples)

