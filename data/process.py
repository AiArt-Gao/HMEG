

import json
import pandas as pd
import numpy as np

import torch


def norm_rel_tag(input_tag):
    if input_tag in ['R', 'Right', 'r', 'HORIZONTAL', 'horizontal', 'HOR']:
        return 'R'
    if input_tag in ['Sub', 'SUB', 'SUBSC']:
        return 'Sub'
    if input_tag in ['Sup', 'SUP', 'SUPER']:
        return 'Sup'
    if input_tag in ['Above', 'ABOVE', 'a', 'A', 'UPPER', 'ULIMIT']:
        return 'Above'
    if input_tag in ['Below', 'BELOW', 'b', 'B', 'UNDER', 'LLIMIT']:
        return 'Below'
    if input_tag in ['I', 'Inside']:
        return 'Inside'
    return input_tag


def fix_tag(x):
    x = str(x)
    x_list = x.split('_')
    if x_list[0] == 'COMMA':
        return '_'.join([',', *x_list[1:]])
    else:
        return x


class LG2Graph(object):

    def __init__(self, object_name_to_idx, pred_name_to_idx):
        self.node_types = object_name_to_idx
        self.edge_types = pred_name_to_idx

    def __call__(self, lg_path):
        obj_csv, rel_csv = self.parse_lg(lg_path)
        graph = self.to_graph(obj_csv, rel_csv)
        return graph

    def parse_lg(self, lg_path):
        lg_csv = pd.read_table(lg_path, delimiter=",", comment='#', header=None,
                               skip_blank_lines=True, skipinitialspace=True)
        lg_csv.iloc[:, 0] = lg_csv.iloc[:, 0].apply(lambda x: 'EO' if x != 'O' else x)
        obj_csv = lg_csv[lg_csv.iloc[:, 0] == 'O'].copy()
        rel_csv = lg_csv[lg_csv.iloc[:, 0] == 'EO'].copy()
        obj_csv.columns = ['mark', 'id', 'node_type', 'weight', 'path']
        rel_csv.columns = ['mark', 'from_node', 'to_node', 'edge_type', 'weight']
        obj_csv.iloc[:, 1] = obj_csv.iloc[:, 1].apply(fix_tag)

        id_map = {}
        for i, tag in enumerate(obj_csv.iloc[:, 1]):
            id_map[tag] = i

        obj_csv = obj_csv.sort_values(by=['id'], key=lambda x: x.map(id_map))
        obj_csv.iloc[:, 1] = obj_csv.iloc[:, 1].apply(lambda x: id_map[x])
        obj_csv.iloc[:, 2] = obj_csv.iloc[:, 2].apply(fix_tag)
        obj_csv.iloc[:, 2] = obj_csv.iloc[:, 2].apply(lambda x: self.node_types[str(x)])
        rel_csv.iloc[:, 1] = rel_csv.iloc[:, 1].apply(fix_tag)
        rel_csv.iloc[:, 2] = rel_csv.iloc[:, 2].apply(fix_tag)
        rel_csv.iloc[:, 1] = rel_csv.iloc[:, 1].apply(lambda x: id_map[x])
        rel_csv.iloc[:, 2] = rel_csv.iloc[:, 2].apply(lambda x: id_map[x])
        rel_csv.iloc[:, 3] = rel_csv.iloc[:, 3].apply(lambda x: self.edge_types[norm_rel_tag(x)])
        return obj_csv, rel_csv

    def to_graph(self, obj_csv, rel_csv):
        node_types = obj_csv['node_type'].to_numpy()
        from_node = rel_csv['from_node'].to_numpy().reshape(1, -1)
        to_node = rel_csv['to_node'].to_numpy().reshape(1, -1)
        edge_types = rel_csv['edge_type'].to_numpy()
        edges = np.concatenate([from_node, to_node], axis=0)
        return {
            'node_types': node_types,
            'edge_types': edge_types,
            'edges': edges
        }


class CROHME2Graph(object):

    def __init__(self, vocab):
        self.object_name_to_idx = vocab['object_name_to_idx']
        self.object_idx_to_name = vocab['object_idx_to_name']
        self.pred_name_to_idx = vocab['pred_name_to_idx']
        self.pred_idx_to_name = vocab['pred_idx_to_name']
        self.lg2graph = LG2Graph(self.object_name_to_idx, self.pred_name_to_idx)

    def convert(self, lg_path):
        graph = self.lg2graph(lg_path)
        edge_types = self._encode_edge_type(graph)
        objs = torch.tensor(graph['node_types']).long()
        n = objs.shape[0]
        triples = []
        for row in range(n):
            for col in range(n):
                triples.append([row, edge_types[row, col], col])
        triples = torch.LongTensor(triples)
        return objs, triples

    def _encode_edge_type(self, graph):
        n_nodes = len(graph['node_types'])
        g = torch.zeros(n_nodes * n_nodes, dtype=torch.long)
        edges = graph['edges']
        for i, e in enumerate(graph['edge_types']):
            et = e
            g[edges[0, i] * n_nodes + edges[1, i]] = et
            if e % 2:
                g[edges[1, i] * n_nodes + edges[0, i]] = et + 1
            else:
                g[edges[1, i] * n_nodes + edges[0, i]] = max(et - 1, 0)
        return g.reshape(n_nodes, n_nodes)













