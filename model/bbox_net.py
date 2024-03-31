import torch
import torch.nn as nn
import torch.nn.functional as F

from model.graph import GraphTripleConv, GraphTripleConvNet
from model.layers import build_mlp


class BBoxNet(nn.Module):

    def __init__(self, vocab, embedding_dim=128,
                 gconv_dim=128, gconv_hidden_dim=512,
                 gconv_pooling='avg', gconv_num_layers=5,
                 mlp_normalization='none'):
        super(BBoxNet, self).__init__()

        self.vocab = vocab
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
        box_net_layers = [gconv_dim * 2 + 64, gconv_hidden_dim, box_net_dim]
        self.box_net = build_mlp(box_net_layers, batch_norm=mlp_normalization)
        self.graph_encoder = AttentionModule(gconv_dim)
        # self.aggregator = GraphAggregator(node_hidden_sizes=[128],
        #                                   graph_transform_sizes=[128],
        #                                   input_size=[4], gated=True)
        self.apply(weight_init)

    def forward(self, objs, triples, noise, obj_to_img=None, ):
        O, T = objs.size(0), triples.size(0)
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        if obj_to_img is None:
            obj_to_img = torch.zeros(O, dtype=objs.dtype, device=objs.device)

        obj_vecs = self.obj_embeddings(objs)
        pred_vecs = self.pred_embeddings(p)

        if isinstance(self.gconv, nn.Linear):
            obj_vecs = self.gconv(obj_vecs)
        else:
            obj_vecs, pred_vecs = self.gconv(obj_vecs, pred_vecs, edges)
        if self.gconv_net is not None:
            obj_vecs, pred_vecs = self.gconv_net(obj_vecs, pred_vecs, edges)
        graph_vecs = self.graph_encoder(obj_vecs, obj_to_img)

        vecs = torch.cat([obj_vecs, graph_vecs, noise], dim=1)
        boxes_pred = self.box_net(vecs)
        return boxes_pred


class GraphAggregator(nn.Module):
    """This module computes graph representations by aggregating from parts."""

    def __init__(self,
                 node_hidden_sizes,
                 graph_transform_sizes=None,
                 input_size=None,
                 gated=True,
                 aggregation_type='sum',
                 name='graph-aggregator'):
        """Constructor.

        Args:
          node_hidden_sizes: the hidden layer sizes of the node transformation nets.
            The last element is the size of the aggregated graph representation.

          graph_transform_sizes: sizes of the transformation layers on top of the
            graph representations.  The last element of this list is the final
            dimensionality of the output graph representations.

          gated: set to True to do gated aggregation, False not to.

          aggregation_type: one of {sum, max, mean, sqrt_n}.
          name: name of this module.
        """
        super(GraphAggregator, self).__init__()

        self._node_hidden_sizes = node_hidden_sizes
        self._graph_transform_sizes = graph_transform_sizes
        self._graph_state_dim = node_hidden_sizes[-1]
        self._input_size = input_size
        #  The last element is the size of the aggregated graph representation.
        self._gated = gated
        self._aggregation_type = aggregation_type
        self._aggregation_op = None
        self.MLP1, self.MLP2 = self.build_model()

    def build_model(self):
        node_hidden_sizes = self._node_hidden_sizes
        if self._gated:
            node_hidden_sizes[-1] = self._graph_state_dim * 2

        layer = []
        layer.append(nn.Linear(self._input_size[0], node_hidden_sizes[0]))
        for i in range(1, len(node_hidden_sizes)):
            layer.append(nn.LeakyReLU())
            layer.append(nn.Linear(node_hidden_sizes[i - 1], node_hidden_sizes[i]))
        MLP1 = nn.Sequential(*layer)

        if (self._graph_transform_sizes is not None and
                len(self._graph_transform_sizes) > 0):
            layer = []
            layer.append(nn.Linear(self._graph_state_dim, self._graph_transform_sizes[0]))
            for i in range(1, len(self._graph_transform_sizes)):
                layer.append(nn.LeakyReLU())
                layer.append(nn.Linear(self._graph_transform_sizes[i - 1], self._graph_transform_sizes[i]))
            MLP2 = nn.Sequential(*layer)

        return MLP1, MLP2

    def forward(self, node_states, graph_idx, n_graphs):
        """Compute aggregated graph representations.

        Args:
          node_states: [n_nodes, node_state_dim] float tensor, node states of a
            batch of graphs concatenated together along the first dimension.
          graph_idx: [n_nodes] int tensor, graph ID for each node.
          n_graphs: integer, number of graphs in this batch.

        Returns:
          graph_states: [n_graphs, graph_state_dim] float tensor, graph
            representations, one row for each graph.
        """

        node_states_g = self.MLP1(node_states)

        if self._gated:
            gates = torch.sigmoid(node_states_g[:, :self._graph_state_dim])
            node_states_g = node_states_g[:, self._graph_state_dim:] * gates

        graph_states = unsorted_segment_sum(node_states_g, graph_idx, n_graphs)

        if self._aggregation_type == 'max':
            # reset everything that's smaller than -1e5 to 0.
            graph_states *= torch.FloatTensor(graph_states > -1e5)
        # transform the reduced graph states further
        if (self._graph_transform_sizes is not None and
                len(self._graph_transform_sizes) > 0):
            graph_states = self.MLP2(graph_states)

        return graph_states


def unsorted_segment_sum(data, segment_ids, num_segments, device='cuda'):
    """
    Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

    :param data: A tensor whose segments are to be summed.
    :param segment_ids: The segment indices tensor.
    :param num_segments: The number of segments.
    :return: A tensor of same data type as the data argument.
    """
    assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"
    # Encourage to use the below code when a deterministic result is
    # needed (reproducibility). However, the code below is with low efficiency.
    # tensor = torch.zeros(num_segments, data.shape[1]).cuda()
    # for index in range(num_segments):
    #     tensor[index, :] = torch.sum(data[segment_ids == index, :], dim=0)
    # return tensor
    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(data.shape[1:])).long().to(device)
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])
    assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"
    shape = [num_segments] + list(data.shape[1:])
    tensor = torch.zeros(*shape).to(device).scatter_add(0, segment_ids, data)
    tensor = tensor.type(data.dtype)
    return tensor


class AttentionModule(torch.nn.Module):

    def __init__(self, dim):
        super(AttentionModule, self).__init__()
        self.dim = dim
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.dim, self.dim))

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding, obj_to_img):
        representations = []
        N = obj_to_img.data.max().item() + 1
        for i in range(N):
            emb = embedding[obj_to_img == i]
            O, D = emb.shape
            global_context = torch.mean(torch.matmul(emb, self.weight_matrix), dim=0)
            transformed_global = torch.tanh(global_context)
            sigmoid_scores = torch.sigmoid(torch.mm(emb, transformed_global.view(-1, 1)))
            representation = torch.mm(torch.t(emb), sigmoid_scores).reshape(1, -1)
            representation = representation.repeat_interleave(O, 0)
            representations.append(representation)
        return torch.cat(representations, dim=0)


class AttentionModule2(torch.nn.Module):

    def __init__(self, dim):
        super(AttentionModule2, self).__init__()
        self.dim = dim
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.dim, self.dim))

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding, obj_to_img):
        representations = []
        N = obj_to_img.data.max().item() + 1
        for i in range(N):
            emb = embedding[obj_to_img == i]
            O, D = emb.shape
            global_context = torch.mean(torch.matmul(emb, self.weight_matrix), dim=0)
            transformed_global = torch.tanh(global_context)
            sigmoid_scores = torch.sigmoid(torch.mm(emb, transformed_global.view(-1, 1)))
            representation = torch.mm(torch.t(emb), sigmoid_scores).reshape(1, -1)
            representations.append(representation)
        return torch.cat(representations, dim=0)


class LayoutEncoder(nn.Module):

    def __init__(self, in_dim, hidden_dims):
        super(LayoutEncoder, self).__init__()
        self._in_dim = in_dim
        self._hidden_dims = hidden_dims
        self.mlp = self._build_model()
        self.encoder = AttentionModule2(self._hidden_dims[-1])

    def _build_model(self):
        layer = list()
        layer.append(nn.Linear(self._in_dim, self._hidden_dims[0]))
        for i in range(1, len(self._hidden_dims)):
            layer.append(nn.LeakyReLU())
            layer.append(nn.BatchNorm1d(self._hidden_dims[i - 1]))
            layer.append(nn.Linear(self._hidden_dims[i - 1], self._hidden_dims[i]))
        return nn.Sequential(*layer)

    def forward(self, node_features):
        layout_embedding = self.mlp(node_features)
        latent = self.encoder(layout_embedding).reshape(1, -1)
        return latent


class LayoutDiscriminator(nn.Module):

    def __init__(self, vocab):
        super(LayoutDiscriminator, self).__init__()
        self.vocab = vocab
        num_objs = len(vocab['object_idx_to_name'])
        num_preds = len(vocab['pred_idx_to_name'])
        self.obj_embeddings = nn.Embedding(num_objs + 1, 32)
        self.pred_embeddings = nn.Embedding(num_preds, 32)

        self.model = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.Linear(16, 16),
            # nn.BatchNorm1d(16),
            # nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.linear = nn.Sequential(
            nn.Linear(32 + 4, 32),
            # nn.Linear(4, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        gconv_kwargs = {
            'input_dim': 32,
            'output_dim': 32,
            'hidden_dim': 32,
            'pooling': 'avg',
        }
        self.gconv = GraphTripleConv(**gconv_kwargs)
        self.gconv_net = None
        gconv_kwargs = {
            'input_dim': 32,
            'hidden_dim': 32,
            'pooling': 'avg',
            'num_layers': 1,
        }
        self.gconv_net = GraphTripleConvNet(**gconv_kwargs)
        self.encoder = AttentionModule2(dim=32)

        self.apply(weight_init)

    def forward(self, objs, boxes, triples, obj_to_img):
        boxes = self._normalize(boxes)
        objs_vecs = self.obj_embeddings(objs)
        objs_vecs = torch.cat([objs_vecs, boxes], dim=1)
        # objs_vecs = boxes
        objs_vecs = self.linear(objs_vecs)
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)
        pred_vecs = self.pred_embeddings(p)
        objs_vecs, pred_vecs = self.gconv(objs_vecs, pred_vecs, edges)
        objs_vecs, pred_vecs = self.gconv_net(objs_vecs, pred_vecs, edges)
        graph_embeddings = self.encoder(objs_vecs, obj_to_img)
        validity = self.model(graph_embeddings)
        return validity

    def _normalize(self, boxes):
        with torch.no_grad():
            m = boxes.mean(dim=0)
            s = boxes.std(dim=0)
        return (boxes - m) / (s + 1e-7)


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
         torch.nn.init.xavier_normal_(m.weight.data)
         torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
         torch.nn.init.xavier_normal_(m.weight)
         torch.nn.init.constant_(m.bias, 0.0)




if __name__ == '__main__':
    import os
    import json
    import random

    from data.crohme import CROHMELabelGraphDataset, crohme_collate_fn
    from torch.utils.data import DataLoader

    CROHME_DIR = os.path.expanduser('../../datasets/crohme2019')

    def _build_crohme_dsets(crohme_dir, image_size=(256, 256)):
        with open(os.path.join(crohme_dir, 'vocab.json')) as f:
            vocab = json.load(f)
        nc = len(vocab['object_idx_to_name'])
        npy_dir = os.path.join(crohme_dir, 'link_npy')
        names = [name[:-4] for name in os.listdir(npy_dir)][:600]
        random.shuffle(names)
        train_dset = CROHMELabelGraphDataset(crohme_dir, names[:-500], nc=nc, image_size=image_size)
        val_dset = CROHMELabelGraphDataset(crohme_dir, names[-500:], nc=nc, image_size=image_size)
        return vocab, train_dset, val_dset

    def _build_loaders(crohme_dir):
        vocab, train_dset, val_dset = _build_crohme_dsets(crohme_dir)
        collate_fn = crohme_collate_fn
        loader_kwargs = {
            'batch_size': 6,
            'num_workers': 4,
            'shuffle': True,
            'collate_fn': collate_fn,
        }
        train_loader = DataLoader(train_dset, **loader_kwargs)
        loader_kwargs['shuffle'] = True
        val_loader = DataLoader(val_dset, **loader_kwargs)
        return vocab, train_loader, val_loader



    vocab, train_loader, val_loader = _build_loaders(CROHME_DIR)
    box_net = BBoxNet(vocab=vocab).cuda()
    data_iter = iter(train_loader)
    imgs_64, imgs_128, imgs_256, objs, boxes, layout_matrix, triples, obj_to_img, triple_to_img = next(data_iter)

    print(objs.shape)
    print(triples.shape)
    print(obj_to_img.shape)

    boxes_pred, graph_embeddings = box_net(objs.cuda(), triples.cuda(), obj_to_img.cuda())
    print(boxes_pred.shape)
    print(graph_embeddings.shape)

