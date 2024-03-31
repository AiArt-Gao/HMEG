

from model.generator import Sg2ImModel
from model.discriminators import PatchDiscriminator, AcCropDiscriminator
from model.losses import get_gan_losses
from model.utils import int_tuple, float_tuple, str_tuple
from model.utils import timeit, bool_flag, LossManager
from model.metrics import jaccard
from data.crohme import CROHMELabelGraphDataset, crohme_collate_fn
from data import imagenet_deprocess_batch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import os
import json
import random
import argparse
import math
import numpy as np
import itertools
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='crohme', choices=['vg', 'coco', 'crohme'])

# Optimization hyperparameters
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--num_iterations', default=10000000, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)

# Switch the generator to eval mode after this many iterations
parser.add_argument('--eval_mode_after', default=100000, type=int)

# Dataset options common to both VG and COCO
parser.add_argument('--image_size', default='256,256', type=int_tuple)
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=1024, type=int)
parser.add_argument('--shuffle_val', default=True, type=bool_flag)
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--include_relationships', default=True, type=bool_flag)

# Generator options
parser.add_argument('--mask_size', default=16, type=int)  # Set this to 0 to use no masks
parser.add_argument('--embedding_dim', default=128, type=int)
parser.add_argument('--gconv_dim', default=128, type=int)
parser.add_argument('--gconv_hidden_dim', default=512, type=int)
parser.add_argument('--gconv_num_layers', default=5, type=int)
parser.add_argument('--mlp_normalization', default='none', type=str)
parser.add_argument('--refinement_network_dims', default='1024,512,256,128,64', type=int_tuple)
parser.add_argument('--normalization', default='batch')
parser.add_argument('--activation', default='leakyrelu-0.2')
parser.add_argument('--layout_noise_dim', default=32, type=int)
parser.add_argument('--use_boxes_pred_after', default=-1, type=int)

# Generator losses
parser.add_argument('--mask_loss_weight', default=0, type=float)
parser.add_argument('--l1_pixel_loss_weight', default=1.0, type=float)
parser.add_argument('--bbox_pred_loss_weight', default=10, type=float)
parser.add_argument('--predicate_pred_loss_weight', default=0, type=float)  # DEPRECATED

# Generic discriminator options
parser.add_argument('--discriminator_loss_weight', default=0.01, type=float)
parser.add_argument('--gan_loss_type', default='gan')
parser.add_argument('--d_clip', default=None, type=float)
parser.add_argument('--d_normalization', default='batch')
parser.add_argument('--d_padding', default='valid')
parser.add_argument('--d_activation', default='leakyrelu-0.2')

# Object discriminator
parser.add_argument('--d_obj_arch',
                    default='C4-64-2,C4-128-2,C4-256-2')
parser.add_argument('--crop_size', default=32, type=int)
parser.add_argument('--d_obj_weight', default=1.0, type=float)  # multiplied by d_loss_weight
parser.add_argument('--ac_loss_weight', default=0.1, type=float)

# Image discriminator
parser.add_argument('--d_img_arch',
                    default='C4-64-2,C4-128-2,C4-256-2')
parser.add_argument('--d_img_weight', default=1.0, type=float)  # multiplied by d_loss_weight

# Output options
parser.add_argument('--print_every', default=10, type=int)
parser.add_argument('--timing', default=False, type=bool_flag)
parser.add_argument('--checkpoint_every', default=10000, type=int)
parser.add_argument('--output_dir', default=os.getcwd())
parser.add_argument('--checkpoint_name', default='layout_conv_sum2_new')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=False, type=bool_flag)


CROHME_DIR = os.path.expanduser('../datasets/crohme2019')


class Trainer(object):

    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.writer = SummaryWriter()

        self.vocab, self.train_loader, self.val_loader = self._build_loaders()
        self.generator, self.g_kwargs = self._build_generator(self.vocab)
        self.generator.to(device)
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=args.learning_rate)

        self.obj_discriminators, self.d_obj_kwargs = self._build_obj_discriminators(self.vocab)
        self.img_discriminators, self.d_img_kwargs = self._build_img_discriminators(self.vocab)
        self.g_loss_func, self.d_loss_func = get_gan_losses(args.gan_loss_type)

        if self.obj_discriminators is not None:
            for i in range(len(self.obj_discriminators)):
                self.obj_discriminators[i].to(device)
                self.obj_discriminators[i].train()
            self.optimizer_d_obj = torch.optim.Adam(itertools.chain(*[d.parameters() for d in self.obj_discriminators]),
                                                    lr=args.learning_rate)

        if self.img_discriminators is not None:
            for i in range(len(self.img_discriminators)):
                self.img_discriminators[i].to(device)
                self.img_discriminators[i].train()
            self.optimizer_d_img = torch.optim.Adam(itertools.chain(*[d.parameters() for d in self.img_discriminators]),
                                                    lr=args.learning_rate)

        self.checkpoint = {
            'args': args.__dict__,
            'vocab': self.vocab,
            'model_kwargs': self.g_kwargs,
            'd_obj_kwargs': self.d_obj_kwargs,
            'd_img_kwargs': self.d_img_kwargs,
            'losses_ts': [],
            'losses': defaultdict(list),
            'd_losses': defaultdict(list),
            'checkpoint_ts': [],
            'train_batch_data': [],
            'train_samples': [],
            'train_iou': [],
            'val_batch_data': [],
            'val_samples': [],
            'val_losses': defaultdict(list),
            'val_iou': [],
            'norm_d': [],
            'norm_g': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'model_state': None, 'model_best_state': None, 'optim_state': None,
            'd_obj_state': None, 'd_obj_best_state': None, 'd_obj_optim_state': None,
            'd_img_state': None, 'd_img_best_state': None, 'd_img_optim_state': None,
            'best_t': [],
        }

    def train(self):
        step = 0
        for epoch in range(self.args.num_iterations):
            for t, batch in enumerate(self.train_loader):
                step += 1
                if step == self.args.eval_mode_after:
                    print('switching to eval mode')
                    self.generator.eval()
                    self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.args.learning_rate)

                batch = [tensor.to(self.device) for tensor in batch]
                imgs_64, imgs_128, imgs_256, objs, boxes, layout_matrix, triples, obj_to_img, triple_to_img = batch
                imgs = [imgs_64, imgs_128, imgs_256]
                predicates = triples[:, 1]
                masks = None
                with timeit('forward', self.args.timing):
                    model_boxes = boxes
                    model_masks = masks
                    model_out = self.generator(objs, triples, obj_to_img,
                                               boxes_gt=model_boxes, masks_gt=model_masks)
                    imgs_64_pred, imgs_128_pred, imgs_256_pred, \
                    boxes_pred, masks_pred, predicate_scores, layout_matrix_pred = model_out
                    imgs_pred = [imgs_64_pred, imgs_128_pred, imgs_256_pred]
                with timeit('loss', self.args.timing):
                    # Skip the pixel loss if using GT boxes
                    skip_pixel_loss = (model_boxes is None)
                    total_loss, losses = calculate_model_losses(
                        self.args, skip_pixel_loss, imgs, imgs_pred,
                        boxes, boxes_pred, masks, masks_pred,
                        predicates, predicate_scores)
                    layout_loss = F.mse_loss(layout_matrix_pred, layout_matrix)
                    add_loss(total_loss, layout_loss, losses, 'layout_loss', 1)

                if self.obj_discriminators is not None:
                    for i in range(len(imgs)):
                        scores_fake, ac_loss = self.obj_discriminators[i](imgs_pred[i], objs, boxes, obj_to_img)
                        total_loss = add_loss(total_loss, ac_loss, losses, 'ac_loss_%d' % i,
                                              self.args.ac_loss_weight)
                        weight = self.args.discriminator_loss_weight * self.args.d_obj_weight
                        total_loss = add_loss(total_loss, self.g_loss_func(scores_fake), losses,
                                              'g_gan_obj_loss_%d' % i, weight)
                if self.img_discriminators is not None:
                    for i in range(len(imgs)):
                        scores_fake = self.img_discriminators[i](imgs_pred[i])
                        weight = self.args.discriminator_loss_weight * self.args.d_img_weight
                        total_loss = add_loss(total_loss, self.g_loss_func(scores_fake), losses,
                                              'g_gan_img_loss_%d' % i, weight)
                losses['total_loss'] = total_loss.item()

                if not math.isfinite(losses['total_loss']):
                    print('WARNING: Got loss = NaN, not backpropping')
                    continue
                self.optimizer.zero_grad()
                with timeit('backward', self.args.timing):
                    total_loss.backward()
                self.optimizer.step()

                d_obj_losses, d_img_losses = None, None
                if self.obj_discriminators is not None:
                    d_obj_losses = LossManager()
                    for i in range(len(imgs)):
                        imgs_fake = imgs_pred[i].detach()
                        scores_fake, ac_loss_fake = self.obj_discriminators[i](imgs_fake, objs, boxes, obj_to_img)
                        scores_real, ac_loss_real = self.obj_discriminators[i](imgs[i], objs, boxes, obj_to_img)
                        d_obj_gan_loss = self.d_loss_func(scores_real, scores_fake)
                        d_obj_losses.add_loss(d_obj_gan_loss, 'd_obj_gan_loss_%d' % i)
                        d_obj_losses.add_loss(ac_loss_real, 'd_ac_loss_real_%d' % i)
                        d_obj_losses.add_loss(ac_loss_fake, 'd_ac_loss_fake_%d' % i)
                    self.optimizer_d_obj.zero_grad()
                    d_obj_losses.total_loss.backward()
                    self.optimizer_d_obj.step()

                if self.img_discriminators is not None:
                    d_img_losses = LossManager()
                    for i in range(len(imgs)):
                        imgs_fake = imgs_pred[i].detach()
                        scores_fake = self.img_discriminators[i](imgs_fake)
                        scores_real = self.img_discriminators[i](imgs[i])
                        d_img_gan_loss = self.d_loss_func(scores_real, scores_fake)
                        d_img_losses.add_loss(d_img_gan_loss, 'd_img_gan_loss_%d' % i)
                    self.optimizer_d_img.zero_grad()
                    d_img_losses.total_loss.backward()
                    self.optimizer_d_img.step()

                if step % self.args.print_every == 0:
                    self._record(step, losses, d_obj_losses, d_img_losses)

                if step % self.args.checkpoint_every == 0:
                    self._check(epoch, t)

    def _record(self, step, losses, d_obj_losses, d_img_losses):
        print('t = %d / %d' % (step, self.args.num_iterations))
        for tag, val in losses.items():
            self.writer.add_scalar(tag, val, step)
            self.checkpoint['losses'][tag].append(val)
            print(' G [%s]: %.4f' % (tag, val))
        if d_obj_losses is not None:
            for tag, val in d_obj_losses.items():
                self.writer.add_scalar(tag, val, step)
                self.checkpoint['d_losses'][tag].append(val)
                print(' D_obj [%s]: %.4f' % (tag, val))
        if d_img_losses is not None:
            for tag, val in d_img_losses.items():
                self.writer.add_scalar(tag, val, step)
                self.checkpoint['d_losses'][tag].append(val)
                print(' D_img [%s]: %.4f' % (tag, val))

    def _check(self, epoch, t):
        print('checking on train')
        train_results = self._check_model(self.train_loader)
        t_losses, t_samples, t_batch_data, t_avg_iou = train_results
        self.checkpoint['train_batch_data'].append(t_batch_data)
        self.checkpoint['train_samples'].append(t_samples)
        self.checkpoint['checkpoint_ts'].append(t)
        self.checkpoint['train_iou'].append(t_avg_iou)
        print('checking on val')
        val_results = self._check_model(self.val_loader)
        val_losses, val_samples, val_batch_data, val_avg_iou = val_results
        self.checkpoint['val_samples'].append(val_samples)
        self.checkpoint['val_batch_data'].append(val_batch_data)
        self.checkpoint['val_iou'].append(val_avg_iou)
        print('train iou: ', t_avg_iou)
        print('val iou: ', val_avg_iou)
        for k, v in val_losses.items():
            self.checkpoint['val_losses'][k].append(v)
        self.checkpoint['model_state'] = self.generator.state_dict()
        self.checkpoint['optim_state'] = self.optimizer.state_dict()
        if self.obj_discriminators is not None:
            self.checkpoint['d_obj_state'] = [d.state_dict() for d in self.obj_discriminators]
            self.checkpoint['d_obj_optim_state'] = self.optimizer_d_obj.state_dict()
        if self.img_discriminators is not None:
            self.checkpoint['d_img_state'] = [d.state_dict() for d in self.img_discriminators]
            self.checkpoint['d_img_optim_state'] = self.optimizer_d_img.state_dict()
        self.checkpoint['counters']['t'] = t
        self.checkpoint['counters']['epoch'] = epoch
        checkpoint_path = os.path.join(self.args.output_dir,
                                       '%s_with_model.pt' % self.args.checkpoint_name)
        print('Saving checkpoint to ', checkpoint_path)
        torch.save(self.checkpoint, checkpoint_path)
        checkpoint_path = os.path.join(self.args.output_dir,
                                       '%s_no_model.pt' % self.args.checkpoint_name)
        key_blacklist = ['model_state', 'optim_state', 'model_best_state',
                         'd_obj_state', 'd_obj_optim_state', 'd_obj_best_state',
                         'd_img_state', 'd_img_optim_state', 'd_img_best_state']
        small_checkpoint = {}
        for k, v in self.checkpoint.items():
            if k not in key_blacklist:
                small_checkpoint[k] = v
        torch.save(small_checkpoint, checkpoint_path)

    def _check_model(self, loader):
        num_samples = 0
        all_losses = defaultdict(list)
        total_iou = 0
        total_boxes = 0
        with torch.no_grad():
            for batch in loader:
                batch = [tensor.cuda() for tensor in batch]
                masks = None
                imgs_64, imgs_128, imgs_256, objs, boxes, layout_matrix, triples, obj_to_img, triple_to_img = batch
                imgs = [imgs_64, imgs_128, imgs_256]
                predicates = triples[:, 1]
                # Run the model as it has been run during training
                model_masks = masks
                model_out = self.generator(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=model_masks)
                imgs_64_pred, imgs_128_pred, imgs_256_pred, \
                boxes_pred, masks_pred, predicate_scores, layout_matrix_pred = model_out
                imgs_pred = [imgs_64_pred, imgs_128_pred, imgs_256_pred]
                skip_pixel_loss = False
                total_loss, losses = calculate_model_losses(
                    self.args, skip_pixel_loss, imgs, imgs_pred,
                    boxes, boxes_pred, masks, masks_pred,
                    predicates, predicate_scores)
                layout_loss = F.mse_loss(layout_matrix_pred, layout_matrix)
                add_loss(total_loss, layout_loss, losses, 'layout_loss', 1)
                total_iou += jaccard(boxes_pred, boxes)
                total_boxes += boxes_pred.size(0)
                for loss_name, loss_val in losses.items():
                    all_losses[loss_name].append(loss_val)
                num_samples += imgs[0].size(0)
                if num_samples >= self.args.num_val_samples:
                    break
            samples = {}
            samples['gt_img'] = imgs[2]
            model_out = self.generator(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=masks)
            samples['gt_box_gt_mask'] = model_out[2]
            model_out = self.generator(objs, triples, obj_to_img, boxes_gt=boxes)
            samples['gt_box_pred_mask'] = model_out[2]
            model_out = self.generator(objs, triples, obj_to_img)
            samples['pred_box_pred_mask'] = model_out[2]
            for k, v in samples.items():
                samples[k] = imagenet_deprocess_batch(v)
            mean_losses = {k: np.mean(v) for k, v in all_losses.items()}
            avg_iou = total_iou / total_boxes
            masks_to_store = masks
            if masks_to_store is not None:
                masks_to_store = masks_to_store.data.cpu().clone()
            masks_pred_to_store = masks_pred
            if masks_pred_to_store is not None:
                masks_pred_to_store = masks_pred_to_store.data.cpu().clone()
        batch_data = {
            'objs': objs.detach().cpu().clone(),
            'boxes_gt': boxes.detach().cpu().clone(),
            'masks_gt': masks_to_store,
            'triples': triples.detach().cpu().clone(),
            'obj_to_img': obj_to_img.detach().cpu().clone(),
            'triple_to_img': triple_to_img.detach().cpu().clone(),
            'boxes_pred': boxes_pred.detach().cpu().clone(),
            'masks_pred': masks_pred_to_store
        }
        out = [mean_losses, samples, batch_data, avg_iou]
        return tuple(out)

    def _build_crohme_dsets(self):
        with open(os.path.join(CROHME_DIR, 'vocab.json')) as f:
            vocab = json.load(f)
        nc = len(vocab['object_idx_to_name'])
        npy_dir = os.path.join(CROHME_DIR, 'link_npy')
        names = [name[:-4] for name in os.listdir(npy_dir)]
        random.shuffle(names)
        train_dset = CROHMELabelGraphDataset(CROHME_DIR, names[:-500], nc=nc, image_size=self.args.image_size)
        val_dset = CROHMELabelGraphDataset(CROHME_DIR, names[-500:], nc=nc, image_size=self.args.image_size)
        return vocab, train_dset, val_dset

    def _build_loaders(self):
        vocab, train_dset, val_dset = self._build_crohme_dsets()
        collate_fn = crohme_collate_fn
        loader_kwargs = {
            'batch_size': self.args.batch_size,
            'num_workers': self.args.loader_num_workers,
            'shuffle': True,
            'collate_fn': collate_fn,
        }
        train_loader = DataLoader(train_dset, **loader_kwargs)
        loader_kwargs['shuffle'] = self.args.shuffle_val
        val_loader = DataLoader(val_dset, **loader_kwargs)
        return vocab, train_loader, val_loader

    def _build_generator(self, vocab):
        if self.args.checkpoint_start_from is not None:
            checkpoint = torch.load(self.args.checkpoint_start_from)
            kwargs = checkpoint['model_kwargs']
            model = Sg2ImModel(**kwargs)
            raw_state_dict = checkpoint['model_state']
            state_dict = {}
            for k, v in raw_state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]
                state_dict[k] = v
            model.load_state_dict(state_dict)
        else:
            kwargs = {
                'vocab': vocab,
                'image_size': self.args.image_size,
                'embedding_dim': self.args.embedding_dim,
                'gconv_dim': self.args.gconv_dim,
                'gconv_hidden_dim': self.args.gconv_hidden_dim,
                'gconv_num_layers': self.args.gconv_num_layers,
                'mlp_normalization': self.args.mlp_normalization,
                'refinement_dims': self.args.refinement_network_dims,
                'normalization': self.args.normalization,
                'activation': self.args.activation,
                'mask_size': self.args.mask_size,
                'layout_noise_dim': self.args.layout_noise_dim,
            }
            model = Sg2ImModel(**kwargs)
        return model, kwargs

    def _build_img_discriminators(self, vocab):
        discriminators = None
        d_kwargs = {}
        d_weight = self.args.discriminator_loss_weight
        d_img_weight = self.args.d_img_weight
        if d_weight == 0 or d_img_weight == 0:
            return discriminators, d_kwargs
        d_kwargs = {
            'arch': self.args.d_img_arch,
            'normalization': self.args.d_normalization,
            'activation': self.args.d_activation,
            'padding': self.args.d_padding,
        }
        discriminators = [PatchDiscriminator(**d_kwargs),
                          PatchDiscriminator(**d_kwargs),
                          PatchDiscriminator(**d_kwargs)]
        return discriminators, d_kwargs

    def _build_obj_discriminators(self, vocab):
        discriminators = None
        d_kwargs = {}
        d_weight = self.args.discriminator_loss_weight
        d_obj_weight = self.args.d_obj_weight
        if d_weight == 0 or d_obj_weight == 0:
            return discriminators, d_kwargs
        d_kwargs = {
            'vocab': vocab,
            'arch': self.args.d_obj_arch,
            'normalization': self.args.d_normalization,
            'activation': self.args.d_activation,
            'padding': self.args.d_padding,
            'object_size': self.args.crop_size,
        }
        discriminators = [AcCropDiscriminator(**d_kwargs),
                          AcCropDiscriminator(**d_kwargs),
                          AcCropDiscriminator(**d_kwargs)]
        return discriminators, d_kwargs

    def load(self, restore_path):
        self.checkpoint = torch.load(restore_path)
        self.generator.load_state_dict(self.checkpoint['model_state'])
        self.generator.eval()
        self.optimizer.load_state_dict(self.checkpoint['optim_state'])

        for i in range(3):
            self.obj_discriminators[i].load_state_dict(self.checkpoint['d_obj_state'][i])
        self.optimizer_d_obj.load_state_dict(self.checkpoint['d_obj_optim_state'])
        for i in range(3):
            self.img_discriminators[i].load_state_dict(self.checkpoint['d_img_state'][i])
        self.optimizer_d_img.load_state_dict(self.checkpoint['d_img_optim_state'])
        print('load: %s' % restore_path)


def add_loss(total_loss, curr_loss, loss_dict, loss_name, weight=1):
    curr_loss = curr_loss * weight
    loss_dict[loss_name] = curr_loss.item()
    if total_loss is not None:
        total_loss += curr_loss
    else:
        total_loss = curr_loss
    return total_loss


def calculate_model_losses(args, skip_pixel_loss, imgs, img_preds,
                           bbox, bbox_pred, masks, masks_pred,
                           predicates, predicate_scores):
    total_loss = torch.zeros(1).to(imgs[0])
    losses = {}
    l1_pixel_weight = args.l1_pixel_loss_weight
    if skip_pixel_loss:
        l1_pixel_weight = 0
    l1_pixel_loss = F.l1_loss(img_preds[0], imgs[0]) + \
                    F.l1_loss(img_preds[1], imgs[1]) + \
                    F.l1_loss(img_preds[2], imgs[2])
    total_loss = add_loss(total_loss, l1_pixel_loss, losses, 'L1_pixel_loss',
                          l1_pixel_weight)
    loss_bbox = F.mse_loss(bbox_pred, bbox)
    total_loss = add_loss(total_loss, loss_bbox, losses, 'bbox_pred',
                          args.bbox_pred_loss_weight)
    if args.predicate_pred_loss_weight > 0:
        loss_predicate = F.cross_entropy(predicate_scores, predicates)
        total_loss = add_loss(total_loss, loss_predicate, losses, 'predicate_pred',
                              args.predicate_pred_loss_weight)
    if args.mask_loss_weight > 0 and masks is not None and masks_pred is not None:
        mask_loss = F.binary_cross_entropy(masks_pred, masks.float())
        total_loss = add_loss(total_loss, mask_loss, losses, 'mask_loss',
                              args.mask_loss_weight)
    return total_loss, losses


if __name__ == '__main__':
    torch.cuda.set_device(3)
    args = parser.parse_args()
    trainer = Trainer(args, 'cuda')
    trainer.load('layout_conv_sum2_new_with_model.pt')
    trainer.train()

