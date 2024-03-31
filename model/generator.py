import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.graph import GraphTripleConv, GraphTripleConvNet
from model.crn import RefinementNetwork
from model.layout import boxes_to_layout, masks_to_layout, _boxes_to_grid, _boxes_to_region, _pool_samples
from model.layers import build_mlp

from data.process import CROHME2Graph


class Sg2ImModel(nn.Module):

    def __init__(self, vocab, image_size=(64, 64), embedding_dim=64,
                 gconv_dim=128, gconv_hidden_dim=512,
                 gconv_pooling='avg', gconv_num_layers=5,
                 refinement_dims=(1024, 512, 256, 128, 64),
                 normalization='batch', activation='leakyrelu-0.2',
                 mask_size=None, mlp_normalization='none', layout_noise_dim=0,
                 **kwargs):
        super(Sg2ImModel, self).__init__()

        # We used to have some additional arguments:
        # vec_noise_dim, gconv_mode, box_anchor, decouple_obj_predictions
        if len(kwargs) > 0:
            print('WARNING: Model got unexpected kwargs ', kwargs)

        self.vocab = vocab
        self.image_size = image_size
        self.layout_noise_dim = layout_noise_dim

        num_objs = len(vocab['object_idx_to_name'])
        num_preds = len(vocab['pred_idx_to_name'])
        self.obj_embeddings = nn.Embedding(num_objs + 1, embedding_dim)
        self.pred_embeddings = nn.Embedding(num_preds, embedding_dim)

        if gconv_num_layers == 0:
            self.gconv = nn.Linear(embedding_dim, gconv_dim)
        elif gconv_num_layers > 0:
            gconv_kwargs = {
                'input_dim': embedding_dim,
                'output_dim': gconv_dim,
                'hidden_dim': gconv_hidden_dim,
                'pooling': gconv_pooling,
                'mlp_normalization': mlp_normalization,
            }
            self.gconv = GraphTripleConv(**gconv_kwargs)

        self.gconv_net = None
        if gconv_num_layers > 1:
            gconv_kwargs = {
                'input_dim': gconv_dim,
                'hidden_dim': gconv_hidden_dim,
                'pooling': gconv_pooling,
                'num_layers': gconv_num_layers - 1,
                'mlp_normalization': mlp_normalization,
            }
            self.gconv_net = GraphTripleConvNet(**gconv_kwargs)

        box_net_dim = 4
        box_net_layers = [gconv_dim, gconv_hidden_dim, box_net_dim]
        self.box_net = build_mlp(box_net_layers, batch_norm=mlp_normalization)

        self.mask_net = None
        if mask_size is not None and mask_size > 0:
            self.mask_net = self._build_mask_net(num_objs, gconv_dim, mask_size)

        self.layout_net = Grid2Mask(embedding_dim=embedding_dim)

        rel_aux_layers = [2 * embedding_dim + 8, gconv_hidden_dim, num_preds]
        self.rel_aux_net = build_mlp(rel_aux_layers, batch_norm=mlp_normalization)

        refinement_kwargs = {
            'dims': (gconv_dim + layout_noise_dim,) + refinement_dims,
            'normalization': normalization,
            'activation': activation,
        }
        self.refinement_net = RefinementNetwork(**refinement_kwargs)

        self.crohme2graph = CROHME2Graph(vocab)

    def _build_mask_net(self, num_objs, dim, mask_size):
        output_dim = 1
        layers, cur_size = [], 1
        while cur_size < mask_size:
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers.append(nn.BatchNorm2d(dim))
            layers.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            cur_size *= 2
        if cur_size != mask_size:
            raise ValueError('Mask size must be a power of 2')
        layers.append(nn.Conv2d(dim, output_dim, kernel_size=1))
        return nn.Sequential(*layers)

    def forward(self, objs, triples, obj_to_img=None,
                boxes_gt=None, masks_gt=None):
        """
        Required Inputs:
        - objs: LongTensor of shape (O,) giving categories for all objects
        - triples: LongTensor of shape (T, 3) where triples[t] = [s, p, o]
          means that there is a triple (objs[s], p, objs[o])

        Optional Inputs:
        - obj_to_img: LongTensor of shape (O,) where obj_to_img[o] = i
          means that objects[o] is an object in image i. If not given then
          all objects are assumed to belong to the same image.
        - boxes_gt: FloatTensor of shape (O, 4) giving boxes to use for computing
          the spatial layout; if not given then use predicted boxes.
        """
        O, T = objs.size(0), triples.size(0)
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        if obj_to_img is None:
            obj_to_img = torch.zeros(O, dtype=objs.dtype, device=objs.device)

        obj_vecs = self.obj_embeddings(objs)
        obj_vecs_orig = obj_vecs
        pred_vecs = self.pred_embeddings(p)

        if isinstance(self.gconv, nn.Linear):
            obj_vecs = self.gconv(obj_vecs)
        else:
            obj_vecs, pred_vecs = self.gconv(obj_vecs, pred_vecs, edges)
        if self.gconv_net is not None:
            obj_vecs, pred_vecs = self.gconv_net(obj_vecs, pred_vecs, edges)

        boxes_pred = self.box_net(obj_vecs)

        masks_pred = None
        if self.mask_net is not None:
            mask_scores = self.mask_net(obj_vecs.view(O, -1, 1, 1))
            masks_pred = mask_scores.squeeze(1).sigmoid()

        s_boxes, o_boxes = boxes_pred[s], boxes_pred[o]
        s_vecs, o_vecs = obj_vecs_orig[s], obj_vecs_orig[o]
        rel_aux_input = torch.cat([s_boxes, o_boxes, s_vecs, o_vecs], dim=1)
        rel_scores = self.rel_aux_net(rel_aux_input)

        H, W = self.image_size
        layout_boxes = boxes_pred if boxes_gt is None else boxes_gt
        # print(objs)
        # print(layout_boxes)
        # TODO layout_boxes to grid,
        grid = _boxes_to_region(layout_boxes, 64, 64)
        grid = grid.permute(0, 3, 1, 2)
        # TODO grid 2 layout_matrix
        layout, layout_matrix_32 = self.layout_net(grid, obj_vecs, obj_to_img)
        # print(torch.sum(layout_matrix_32.view(O, -1), dim=-1))
        # print(obj_vecs[5])
        # if masks_pred is None:
        #     layout = boxes_to_layout(obj_vecs, layout_boxes, obj_to_img, H, W)
        # else:
        #     layout_masks = masks_pred if masks_gt is None else masks_gt
        #     layout = masks_to_layout(obj_vecs, layout_boxes, layout_masks,
        #                              obj_to_img, H, W)

        if self.layout_noise_dim > 0:
            N, C, H, W = layout.size()
            noise_shape = (N, self.layout_noise_dim, H, W)
            layout_noise = torch.randn(noise_shape, dtype=layout.dtype,
                                       device=layout.device)
            layout = torch.cat([layout, layout_noise], dim=1)
        img_64, img_128, img_256 = self.refinement_net(layout)
        return img_64, img_128, img_256, boxes_pred, masks_pred, rel_scores, layout_matrix_32

    def encode_scene_graphs(self, scene_graphs):
        """
        Encode one or more scene graphs using this model's vocabulary. Inputs to
        this method are scene graphs represented as dictionaries like the following:

        {
          "objects": ["cat", "dog", "sky"],
          "relationships": [
            [0, "next to", 1],
            [0, "beneath", 2],
            [2, "above", 1],
          ]
        }

        This scene graph has three relationshps: cat next to dog, cat beneath sky,
        and sky above dog.

        Inputs:
        - scene_graphs: A dictionary giving a single scene graph, or a list of
          dictionaries giving a sequence of scene graphs.

        Returns a tuple of LongTensors (objs, triples, obj_to_img) that have the
        same semantics as self.forward. The returned LongTensors will be on the
        same device as the model parameters.
        """
        if isinstance(scene_graphs, dict):
            # We just got a single scene graph, so promote it to a list
            scene_graphs = [scene_graphs]

        objs, triples, obj_to_img = [], [], []
        obj_offset = 0
        for i, sg in enumerate(scene_graphs):
            # Insert dummy __image__ object and __in_image__ relationships
            sg['objects'].append('__image__')
            image_idx = len(sg['objects']) - 1
            for j in range(image_idx):
                sg['relationships'].append([j, '__in_image__', image_idx])

            for obj in sg['objects']:
                obj_idx = self.vocab['object_name_to_idx'].get(obj, None)
                if obj_idx is None:
                    raise ValueError('Object "%s" not in vocab' % obj)
                objs.append(obj_idx)
                obj_to_img.append(i)
            for s, p, o in sg['relationships']:
                pred_idx = self.vocab['pred_name_to_idx'].get(p, None)
                if pred_idx is None:
                    raise ValueError('Relationship "%s" not in vocab' % p)
                triples.append([s + obj_offset, pred_idx, o + obj_offset])
            obj_offset += len(sg['objects'])
        device = next(self.parameters()).device
        objs = torch.tensor(objs, dtype=torch.int64, device=device)
        triples = torch.tensor(triples, dtype=torch.int64, device=device)
        obj_to_img = torch.tensor(obj_to_img, dtype=torch.int64, device=device)
        return objs, triples, obj_to_img

    def forward_json(self, scene_graphs):
        """ Convenience method that combines encode_scene_graphs and forward. """
        objs, triples, obj_to_img = self.encode_scene_graphs(scene_graphs)
        return self.forward(objs, triples, obj_to_img)

    def forward_lg(self, lg_paths):
        if isinstance(lg_paths, dict):
            # We just got a single scene graph, so promote it to a list
            lg_paths = [lg_paths]
        objs, triples, obj_to_img = [], [], []
        obj_offset = 0
        for i, lg_path in enumerate(lg_paths):
            lg_objs, lg_triples = self.crohme2graph.convert(lg_path)
            for lg_obj in lg_objs:
                objs.append(lg_obj)
                obj_to_img.append(i)

            for s, p, o in lg_triples:
                triples.append([s + obj_offset, p, o + obj_offset])
            obj_offset += len(lg_objs)
        device = next(self.parameters()).device
        objs = torch.tensor(objs, dtype=torch.int64, device=device)
        triples = torch.tensor(triples, dtype=torch.int64, device=device)
        obj_to_img = torch.tensor(obj_to_img, dtype=torch.int64, device=device)

        # print(objs)
        # print(triples)
        # print(obj_to_img)
        # print('---------------')

        return self.forward(objs, triples, obj_to_img)


    def forward_lg1(self, lg_path, boxes=None):
        objs, triples, obj_to_img = [], [], []
        obj_offset = 0
        lg_objs, lg_triples = self.crohme2graph.convert(lg_path)
        for lg_obj in lg_objs:
            objs.append(lg_obj)
            obj_to_img.append(0)
        for s, p, o in lg_triples:
            triples.append([s, p, o])
        device = next(self.parameters()).device
        objs = torch.tensor(objs, dtype=torch.int64, device=device)
        triples = torch.tensor(triples, dtype=torch.int64, device=device)
        obj_to_img = torch.tensor(obj_to_img, dtype=torch.int64, device=device)
        return self.forward(objs, triples, obj_to_img, boxes_gt=boxes)




class Grid2Mask(nn.Module):

    def __init__(self, in_dim=4, hid_dim=64, embedding_dim=128):
        super(Grid2Mask, self).__init__()

        output_dim = 1
        layers = []
        layers.append(nn.Conv2d(in_dim, hid_dim, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(hid_dim))
        layers.append(nn.Conv2d(hid_dim, output_dim, kernel_size=3, stride=1, padding=1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, embedding_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(embedding_dim),
            nn.ConvTranspose2d(embedding_dim, embedding_dim, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, grid, obj_vecs, obj_to_img):
        layout_matrix_64 = self.net(grid)
        layout_64 = layout_matrix_to_layout(layout_matrix_64, obj_vecs, obj_to_img)
        # layout_256 = nn.functional.interpolate(layout_64, [256, 256], mode='area')
        layout_256 = self.conv(layout_64)
        return layout_256, layout_matrix_64




def layout_matrix_to_layout(layout_matrix, obj_vecs, obj_to_img):
    N = obj_to_img.data.max().item() + 1
    layouts = []
    for i in range(N):
        idx = (obj_to_img.data == i).nonzero().view(-1)
        if idx.dim() == 0:
            continue
        vecs = obj_vecs[idx]
        matrix = layout_matrix[idx].float()
        # unique_matrix = torch.zeros_like(matrix)
        # V = torch.gather(matrix, 0, torch.max(matrix, dim=0)[1].unsqueeze(0))
        # unique_idx = torch.where(matrix == V)
        # unique_matrix[unique_idx] = matrix[unique_idx]
        # vecs = obj_vecs[idx]
        # layout = unique_matrix.permute(1, 2, 3, 0).matmul(vecs).permute(0, 3, 1, 2)
        layout = matrix.permute(1, 2, 3, 0).matmul(vecs).permute(0, 3, 1, 2)
        layouts.append(layout)
    layouts = torch.cat(layouts, dim=0)
    return layouts


# def layout_matrix_to_layout(layout_matrix, obj_vecs, obj_to_img):
#     O, D = obj_vecs.size()
#     M = layout_matrix.size(-1)
#     N = obj_to_img.data.max().item() + 1
#     layouts = []
#     for i in range(N):
#         idx = (obj_to_img.data == i).nonzero().view(-1)
#         if idx.dim() == 0:
#             continue
#         matrix = layout_matrix[idx].float().expand(D, -1, 1, M, M)
#         matrix = matrix.squeeze(2).permute(1, 0, 2, 3)
#         vecs = obj_vecs[idx].expand(M, M, -1, D)
#         vecs = vecs.permute(2, 3, 0, 1)
#         layout = matrix * vecs
#         layouts.append(torch.sum(layout, dim=0))
#     layouts = torch.cat(layouts, dim=0)
#     return layouts







if __name__ == '__main__':

    ln = LayoutNet(128)
    a = torch.FloatTensor(12, 128)
    print(ln(a).shape)



