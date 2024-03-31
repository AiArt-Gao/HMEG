

from model.generator import Sg2ImModel
from model.bbox_net import BBoxNet
# from model.generator_bak import Sg2ImModel

import torch

import os
from PIL import Image
from tqdm import tqdm
import json
import numpy as np
import pandas as pd

from data.process import CROHME2Graph
from train import parser



class ResultViewer(object):

    def __call__(self, checkpoint_pth, save_dir='eval_results'):
        os.makedirs(save_dir, exist_ok=True)
        checkpoint = torch.load(checkpoint_pth, map_location='cpu')
        val_patchs = self.view_sample(checkpoint['val_samples'])
        print('num val samples: %d' % len(val_patchs))
        for i, patch in enumerate(tqdm(val_patchs)):
            img = Image.fromarray(patch.numpy())
            img.save(os.path.join(save_dir, 'val_%d.png' % i))

    def view_sample(self, samples):
        patch = []
        for sample in samples:
            gt_img = sample['gt_img']
            gt_box_gt_mask = sample['gt_box_gt_mask']
            gt_box_pred_mask = sample['gt_box_pred_mask']
            pred_box_pred_mask = sample['pred_box_pred_mask']
            n_imgs = gt_img.shape[0]
            for i in range(n_imgs):
                im1 = gt_img[i].squeeze()
                im2 = gt_box_gt_mask[i].squeeze()
                im3 = gt_box_pred_mask[i].squeeze()
                im4 = pred_box_pred_mask[i].squeeze()
                im = torch.cat([im1, im2, im3, im4], dim=1)
                im = im.permute(1, 2, 0)
                patch.append(im)
        return patch


CROHME_DIR = os.path.expanduser('../../datasets/crohme2019')


class BoxPredictor(object):

    def __init__(self, args, gen_checkpoint_path, box_checkpoint_path):
        self.device = 'cuda'
        self.args = args
        with open(os.path.join(CROHME_DIR, 'vocab.json')) as f:
            self.vocab = json.load(f)
        gen_checkpoint = torch.load(gen_checkpoint_path, map_location='cuda')
        kwargs = gen_checkpoint['model_kwargs']
        with torch.no_grad():
            self.model = Sg2ImModel(**kwargs)
            raw_state_dict = gen_checkpoint['model_state']
            state_dict = {}
            for k, v in raw_state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]
                state_dict[k] = v
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

        box_checkpoint = torch.load(box_checkpoint_path, map_location='cpu')
        with torch.no_grad():
            self.box_net = BBoxNet(self.vocab)
            raw_state_dict = box_checkpoint['generator']
            state_dict = {}
            for k, v in raw_state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]
                state_dict[k] = v
            self.box_net.load_state_dict(state_dict)
            self.box_net.to(self.device)
            self.box_net.eval()

        self.crohme2graph = CROHME2Graph(self.vocab)

    def predict(self, lg_paths):
        if isinstance(lg_paths, dict):
            # We just got a single scene graph, so promote it to a list
            lg_paths = [lg_paths]
        objs, triples, obj_to_img = [], [], []
        obj_offset = 0
        for i, lg_path in enumerate(lg_paths):
            lg_objs, lg_triples = self.crohme2graph.convert(lg_path)
            if len(lg_objs) <= 1:
                raise
            for lg_obj in lg_objs:
                objs.append(lg_obj)
                obj_to_img.append(i)

            for s, p, o in lg_triples:
                triples.append([s + obj_offset, p, o + obj_offset])
            obj_offset += len(lg_objs)
        device = self.device
        objs = torch.tensor(objs, dtype=torch.int64, device=device)
        triples = torch.tensor(triples, dtype=torch.int64, device=device)
        obj_to_img = torch.tensor(obj_to_img, dtype=torch.int64, device=device)

        noise = gen_rand_noise(objs.size(0), 64, device)
        new_boxes = self.box_net(objs, triples, noise, obj_to_img=obj_to_img)
        old_boxes = self.model(objs, triples, obj_to_img=obj_to_img)[3]

        return old_boxes, new_boxes

    def forward(self, lg_path, boxes_gt=None):
        # TODO: bbox pred
        return self.model.forward_lg1(lg_path, boxes_gt)


def gen_rand_noise(n_node, dim, device):
    noise = torch.randn(n_node, dim)
    noise = noise.to(device)
    return noise


def fix_tag(x):
    x = str(x)
    x_list = x.split('_')
    if x_list[0] == 'COMMA':
        return '_'.join([',', *x_list[1:]])
    else:
        return x


def get_id_map(lg_path):
    lg_csv = pd.read_table(lg_path, delimiter=",", comment='#', header=None,
                           skip_blank_lines=True, skipinitialspace=True)
    lg_csv.iloc[:, 0] = lg_csv.iloc[:, 0].apply(lambda x: 'EO' if x != 'O' else x)
    lg_csv = lg_csv.iloc[:, :5]
    obj_csv = lg_csv[lg_csv.iloc[:, 0] == 'O'].copy()
    rel_csv = lg_csv[lg_csv.iloc[:, 0] == 'EO'].copy()
    obj_csv.columns = ['mark', 'id', 'node_type', 'weight', 'path']
    rel_csv.columns = ['mark', 'from_node', 'to_node', 'edge_type', 'weight']
    obj_csv.iloc[:, 1] = obj_csv.iloc[:, 1].apply(fix_tag)

    id_map = {}
    for i, tag in enumerate(obj_csv.iloc[:, 1]):
        id_map[tag] = i
    return id_map


def parse_gt_box_json(json_path, lg_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    id_map = get_id_map(lg_path)
    data = sorted(data, key=lambda x: id_map[x['href']])

    bbox = []
    for d in data:
        bbox.append(d['bbox'])
    bbox = np.asarray(bbox) / 300
    return bbox


def parse_pred_box_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # print(data)
    nosl_pred_box = np.asarray(data['no_struct_loss'])
    pred_box = np.asarray(data['struct_loss'])
    return nosl_pred_box, pred_box


def iou_match_grid(box1, box2):
    x11, y11, x12, y12 = np.split(box1, 4, axis=1)
    x21, y21, x22, y22 = np.split(box2, 4, axis=1)
    xa = np.maximum(x11, x21)
    xb = np.minimum(x12, x22)
    ya = np.maximum(y11, y21)
    yb = np.minimum(y12, y22)
    area_inter = np.maximum(0, (xb - xa + 1)) * np.maximum(0, (yb - ya + 1))
    area_1 = (np.abs(x12 - x11) + 1) * (np.abs(y12 - y11) + 1)
    area_2 = (np.abs(x22 - x21) + 1) * (np.abs(y22 - y21) + 1)
    area_union = area_1 + area_2 - area_inter
    iou = area_inter / area_union
    return iou


def dice_match_grid(box1, box2):
    x11, y11, x12, y12 = np.split(box1, 4, axis=1)
    x21, y21, x22, y22 = np.split(box2, 4, axis=1)
    xa = np.maximum(x11, x21)
    xb = np.minimum(x12, x22)
    ya = np.maximum(y11, y21)
    yb = np.minimum(y12, y22)
    area_inter = np.maximum(0, (xb - xa + 1)) * np.maximum(0, (yb - ya + 1))
    area_1 = (np.abs(x12 - x11) + 1) * (np.abs(y12 - y11) + 1)
    area_2 = (np.abs(x22 - x21) + 1) * (np.abs(y22 - y21) + 1)
    area_sum = area_1 + area_2
    dice = (2. * area_inter + 1) / (area_sum + 1)
    return dice



def view_box(boxes, img_size=(300, 300)):
    boxes = torch.from_numpy(boxes)
    N = boxes.size(0)
    x0, y0, x1, y1 = boxes.split(1, 1)
    # x0 = torch.round(x0 * img_size[1]).type(torch.long)
    # y0 = torch.round(y0 * img_size[0]).type(torch.long)
    # x1 = torch.round(x1 * img_size[1]).type(torch.long)
    # y1 = torch.round(y1 * img_size[0]).type(torch.long)
    layout = torch.zeros(*img_size, dtype=torch.uint8)
    for i in range(N):
        layout[y0[i]: y1[i], x0[i]: x1[i]] += 75
    return layout


def compute_avg_iou(gt_dir, pred_dir, lg_dir):
    nosl_iou = 0
    nosl_num_bbox = 0
    iou = 0
    num_bbox = 0
    for file in tqdm(os.listdir(gt_dir)):
        gt_json = os.path.join(gt_dir, file)
        pred_json = os.path.join(pred_dir, file)
        lg = os.path.join(lg_dir, file.split('.')[0] + '.lg')

        if os.path.exists(pred_json):
            try:
                gt_bbox = parse_gt_box_json(gt_json, lg)
                pred_bbox_nosl, pred_bbox = parse_pred_box_json(pred_json)

                gt_bbox = (gt_bbox * 300).astype(np.int32)
                pred_bbox_nosl = (pred_bbox_nosl * 300).astype(np.int32)
                pred_bbox = (pred_bbox * 300).astype(np.int32)
                nosl_result = iou_match_grid(gt_bbox, pred_bbox_nosl)
                result = iou_match_grid(gt_bbox, pred_bbox)
                nosl_iou += np.sum(nosl_result)
                nosl_num_bbox += len(nosl_result)
                iou += np.sum(result)
                num_bbox += len(result)
            except:
                print(file)
    return nosl_iou / nosl_num_bbox, iou / num_bbox


def get_boxes_json(lg_dir):
    args = parser.parse_args()
    p = BoxPredictor(args=args,
                     gen_checkpoint_path='../weight/grid_sample_with_model.pt',
                     box_checkpoint_path='../weight/box_net_40000.pkl')
    # lg_dir = '../test_symlgs'
    json_save_dir = lg_dir + '_nomask_box_json'
    os.makedirs(json_save_dir, exist_ok=True)
    for idx, symlg in enumerate(tqdm(os.listdir(lg_dir))):
        try:
            path = os.path.join(lg_dir, symlg)
            old_boxes, new_boxes = p.predict([path])
            old_boxes = old_boxes.detach().cpu().numpy()
            new_boxes = new_boxes.detach().cpu().numpy()
            box_dict = {
                'no_struct_loss': old_boxes.tolist(),
                'struct_loss': new_boxes.tolist()
            }
            with open(os.path.join(json_save_dir, symlg.split('.')[0] + '.json'), 'w') as f:
                json.dump(box_dict, f, indent=2)
        except Exception as e:
            pass


if __name__ == '__main__':

    gt_dir = '../test_box_data/gt_box_json'
    pred_dir = '../test_box_data/test_symlgs_nomask_box_json'
    lg_dir = '../test_box_data/test_symlgs'
    print(compute_avg_iou(gt_dir, pred_dir, lg_dir))

    # gt_dir = '../train_box_data/gt_box_json'
    # pred_dir = '../train_box_data/pred_box_json'
    # lg_dir = '../train_box_data/Train_symlg'
    # print(compute_avg_iou(gt_dir, pred_dir, lg_dir))

    # get_boxes_json('../test_box_data/test_symlgs')



