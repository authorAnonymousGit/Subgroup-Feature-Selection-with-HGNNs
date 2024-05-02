import torch
import torch_geometric
from torch_geometric.data import Data, HeteroData
from itertools import combinations
import pandas as pd
from collections import defaultdict
from sklearn.metrics import mutual_info_score
from utils import *
from config import LatticeGeneration, GNN
import argparse
import tqdm
import warnings
import time
import multiprocessing
from functools import partial

warnings.filterwarnings('ignore')


class FeatureLatticeGraph:
    def __init__(self, dataset_path, args, df=None, create_edges=True, baseline=None):
        self.with_edge_attrs = args.with_edge_attrs
        self.tqdm = args.print_tqdm
        self.create_edges = create_edges
        self.baseline = baseline
        self.dataset = self._read_dataset(dataset_path, df)
        self.feature_num = self.dataset.shape[1] - 2
        self.min_level = get_min_level(args.min_m, args.num_layers)
        self.max_level = get_max_level(args.max_m, args.num_layers, self.feature_num)
        self.edge_sampling_ratio = args.edge_sampling_ratio
        self.subgroups_num = self.dataset['subgroup'].nunique()
        self.cores_to_use = min(self.subgroups_num, int(multiprocessing.cpu_count()))
        self.restricted_graph_idxs_mapping = get_restricted_graph_idxs_mapping(self.feature_num, self.min_level,
                                                                               self.max_level)
        self.mappings_dict = self._create_mappings_dict()
        self.graph = self._create_multiple_feature_lattice()
        self.save(dataset_path)

    @staticmethod
    def _read_dataset(dataset_path, df):
        if df is None:
            df = pd.read_pickle(dataset_path)
        return df

    def _create_mappings_dict(self):
        print(f"Generating the mappings dictionary...\n =====================================\n")
        start = time.time()
        dataframe = self.dataset.astype(str)
        y_series = dataframe['y'].copy()
        with multiprocessing.Pool(processes=self.cores_to_use) as pool:
            mappings_list = list(pool.imap(partial(self._process_gid_mapping_dict, dataframe=dataframe,
                                                   y_series=y_series), range(self.subgroups_num)))
        end = time.time()
        print(f"Mapping dictionary generation time: {round(end - start, 4)} seconds\n ========================\n")
        mappings_dict = self._convert_mappings_list_to_dict(mappings_list)
        return mappings_dict

    @staticmethod
    def _convert_mappings_list_to_dict(mappings_list):
        mappings_dict = dict()
        for mappings in mappings_list:
            mappings_dict.update(mappings)
        return mappings_dict

    def _process_gid_mapping_dict(self, gid, dataframe, y_series):
        mappings_dict, prev_tmp_dict = self._initialize_tmp_dict(gid, dataframe, y_series)
        for comb_size in range(self.min_level + 1, self.max_level + 1):
            mappings_dict, prev_tmp_dict = self._create_comb_size_mappings_dict(gid, mappings_dict, comb_size,
                                                                                dataframe, y_series, prev_tmp_dict)
        return mappings_dict

    def _create_comb_size_mappings_dict(self, gid, mappings_dict, comb_size, dataframe, y_series, prev_tmp_dict):
        mappings_dict[gid][comb_size] = defaultdict(dict)
        feature_set_combs = list(combinations(dataframe.drop(['y', 'subgroup'], axis=1).columns, comb_size))
        # rel_idxs = dataframe[dataframe['subgroup'] == str(gid)].index
        rel_idxs = dataframe[dataframe['subgroup'].apply(lambda x: int(float(x))) == gid].index
        y_series = y_series[rel_idxs]
        comb_property_list = []
        for comb in tqdm.tqdm(feature_set_combs) if self.tqdm else feature_set_combs:
            comb_property_list.append(self._process_comb(comb, rel_idxs, dataframe, prev_tmp_dict, y_series))
        mappings_dict, prev_tmp_dict = self._update_comb_dicts(gid, mappings_dict, comb_size, comb_property_list)
        return mappings_dict, prev_tmp_dict

    def _process_comb(self, comb, rel_idxs, dataframe, prev_tmp_dict, y_series):
        if len(comb) > 1:
            tmp_series = prev_tmp_dict[comb[:-1]] + dataframe[comb[-1]][rel_idxs]
        else:
            tmp_series = prev_tmp_dict[comb].copy()
        score = round(mutual_info_score(tmp_series, y_series), 6)
        binary_vec = convert_comb_to_binary(comb, self.feature_num)
        node_id = convert_binary_to_decimal(binary_vec) - 1
        restricted_node_id = self.restricted_graph_idxs_mapping[node_id]
        return {comb: {'score': score, 'binary_vector': binary_vec, 'node_id': node_id,
                       'tmp_series': tmp_series, 'restricted_node_id': restricted_node_id}}

    @staticmethod
    def _update_comb_dicts(gid, mappings_dict, comb_size, comb_property_list):
        new_tmp_dict = dict()
        for comb_dict in comb_property_list:
            comb = list(comb_dict.keys())[0]
            mappings_dict[gid][comb_size][comb]['score'] = comb_dict[comb]['score']
            mappings_dict[gid][comb_size][comb]['binary_vector'] = comb_dict[comb]['binary_vector']
            mappings_dict[gid][comb_size][comb]['node_id'] = comb_dict[comb]['node_id']
            mappings_dict[gid][comb_size][comb]['restricted_node_id'] = comb_dict[comb]['restricted_node_id']
            new_tmp_dict[comb] = comb_dict[comb]['tmp_series']
        return mappings_dict, new_tmp_dict

    def _initialize_tmp_dict(self, gid, dataframe, y_series):
        mappings_dict = {gid: {1: defaultdict(dict)}}
        prev_tmp_dict = dict()
        feature_set_combs = list(combinations(dataframe.drop(['y', 'subgroup'], axis=1).columns, self.min_level))
        # rel_idxs = dataframe[dataframe['subgroup'] == str(gid)].index
        rel_idxs = dataframe[dataframe['subgroup'].apply(lambda x: int(float(x))) == gid].index
        y_series = y_series[rel_idxs]
        comb_property_list = []
        for comb in feature_set_combs:
            tmp_series = dataframe[comb[0]][rel_idxs]
            prev_tmp_dict[comb] = tmp_series.copy()
            comb_property_list.append(self._process_comb(comb, rel_idxs, dataframe, prev_tmp_dict, y_series))
        mappings_dict, _ = self._update_comb_dicts(gid, mappings_dict, 1, comb_property_list)
        return mappings_dict, prev_tmp_dict

    def _create_multiple_feature_lattice(self):
        print(f"\nCreating the feature lattice...")
        start_time = time.time()
        data = HeteroData()
        data = self._get_node_features_and_labels(data)
        data = self._get_edge_index(data)
        data = self._get_edge_attrs(data)
        end_time = time.time()
        print(f"Feature lattice creation took: {round(end_time - start_time, 4)} seconds\n ========================\n")
        return data

    def _precompute_MI(self):
        print(f"\nComputing MI for lattice...")
        start_time = time.time()
        data = HeteroData()
        data = self._get_node_features_and_labels(data)
        end_time = time.time()
        print(f"Computing MI took: {round(end_time - start_time, 4)} seconds\n ========================\n")
        return data

    def _get_node_features_and_labels(self, data):
        print(f"Getting the node features and labels...\n ------------------\n")
        lattice_nodes_num = get_lattice_nodes_num(self.feature_num, self.min_level, self.max_level)
        with multiprocessing.Pool(processes=self.cores_to_use) as pool:
            gid_nodes_dict_list = list(pool.imap(partial(self._get_gid_nodes, lattice_nodes_num=lattice_nodes_num),
                                                 range(self.subgroups_num)))
        return self._convert_gid_nodes_to_data(gid_nodes_dict_list, data)

    def _get_gid_nodes(self, gid, lattice_nodes_num):
        x_tensor = torch.zeros(len(self.restricted_graph_idxs_mapping), self.feature_num, dtype=torch.float)
        y_tensor = torch.zeros(len(self.restricted_graph_idxs_mapping), dtype=torch.float)
        for comb_size in range(self.min_level, self.max_level + 1):
            combs = self.mappings_dict[gid][comb_size].keys()
            for comb in tqdm.tqdm(combs) if self.tqdm else combs:
                node_id = self.mappings_dict[gid][comb_size][comb]['restricted_node_id']
                x_tensor[node_id] = torch.tensor([int(digit) for digit in
                                                  self.mappings_dict[gid][comb_size][comb]['binary_vector']])
                y_tensor[node_id] = self.mappings_dict[gid][comb_size][comb]['score']
        return {f"g{gid}": {"x": x_tensor, "y": y_tensor}}

    @staticmethod
    def _convert_gid_nodes_to_data(gid_nodes_dict_list, data):
        for gid_nodes_dict in gid_nodes_dict_list:
            gid = list(gid_nodes_dict.keys())[0]
            data[gid].x = gid_nodes_dict[gid]["x"]
            data[gid].y = gid_nodes_dict[gid]["y"]
        return data

    def _get_edge_index(self, data):
        if not self.create_edges:
            return data
        data = self._get_intra_lattice_edges(data)
        data = self._get_inter_lattice_edges(data)
        return data

    def _get_intra_lattice_edges(self, data):
        edge_index = self._get_inter_level_edges()
        edge_index = self._get_intra_level_edges(edge_index)
        for gid in range(self.subgroups_num):
            edge_name = self._get_edge_name(gid, gid)
            data[f"g{gid}", edge_name, f"g{gid}"].edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        return data

    def _get_inter_level_edges(self):
        print(f"Getting the inter-level edges...\n ------------------\n")
        edge_index = []
        gid = 0
        for comb_size in range(self.min_level, self.max_level):
            comb_size_mapping = self.mappings_dict[gid][comb_size]
            next_comb_size_mapping = self.mappings_dict[gid][comb_size + 1]
            for comb, comb_info in comb_size_mapping.items():
                comb_set = set(comb)
                node_id = comb_info['restricted_node_id']
                for next_comb, next_comb_info in next_comb_size_mapping.items():
                    if comb_set.issubset(set(next_comb)):
                        next_node_id = next_comb_info['restricted_node_id']
                        edge_index.extend([[node_id, next_node_id], [next_node_id, node_id]])
        return edge_index

    @staticmethod
    def _convert_gid_inter_edges_to_data(gid_inter_edges_dict_list, data):
        for gid_inter_edges_dict in gid_inter_edges_dict_list:
            gid = list(gid_inter_edges_dict.keys())[0]
            edge_name = gid_inter_edges_dict[gid]['edge_name']
            edge_index = gid_inter_edges_dict[gid]['edge_index']
            data[f"{gid}", edge_name, f"{gid}"].edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        return data

    @staticmethod
    def _get_edge_name(gid1, gid2):
        g_str1 = ''.join(('g', str(gid1)))
        g_str2 = ''.join(('g', str(gid2)))
        return ''.join((g_str1, 'TO', g_str2))

    def _get_intra_level_edges(self, edge_index):
        print(f"Getting the intra-level edges...\n ------------------\n")
        gid = 0
        edge_set = set()
        for comb_size in range(max(2, self.min_level), self.max_level + 1):
            comb_size_mapping = self.mappings_dict[gid][comb_size]
            for comb, comb_info in comb_size_mapping.items():
                node_id = comb_info['restricted_node_id']
                comb_set = set(comb)
                for next_comb, next_comb_info in comb_size_mapping.items():
                    if np.random.rand() > self.edge_sampling_ratio:
                        continue
                    if len(comb_set.intersection(set(next_comb))) == comb_size - 1:
                        if (next_comb, comb) in edge_set:
                            continue
                        edge_set.update([(comb, next_comb), (next_comb, comb)])
                        next_node_id = next_comb_info['restricted_node_id']
                        edge_index.extend([[node_id, next_node_id], [next_node_id, node_id]])
        return edge_index

    @staticmethod
    def _convert_gid_intra_edges_to_data(gid_intra_edges_dict_list, data):
        for gid_intra_edges_dict in gid_intra_edges_dict_list:
            gid = list(gid_intra_edges_dict.keys())[0]
            edge_name = gid_intra_edges_dict[gid]['edge_name']
            edge_index = gid_intra_edges_dict[gid]['edge_index']
            data[f"{gid}", edge_name, f"{gid}"].edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        return data

    def _get_inter_lattice_edges(self, data):
        lattice_nodes_num = get_lattice_nodes_num(self.feature_num, self.min_level, self.max_level)
        edge_index = [[node_id, node_id] for node_id in range(lattice_nodes_num)]
        for gid1 in range(self.subgroups_num):
            for gid2 in range(gid1 + 1, self.subgroups_num):
                edge_name1 = self._get_edge_name(gid1, gid2)
                edge_name2 = self._get_edge_name(gid2, gid1)
                data[f"g{gid1}", edge_name1, f"g{gid2}"].edge_index = torch.tensor(edge_index, dtype=torch.long).t()
                data[f"g{gid2}", edge_name2, f"g{gid1}"].edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        return data

    def _get_edge_attrs(self, data):
        if self.with_edge_attrs:
            # TODO: Implement this method
            pass
        else:
            return data

    def save(self, dataset_path) -> None:
        if self.baseline is not None:
            graph_path = dataset_path.replace('.pkl', f'{self.baseline}_lattice.pt')
        else:
            graph_path = dataset_path.replace('.pkl', f'_hetero_graph_edgeSamplingRatio={self.edge_sampling_ratio}.pt')
        torch.save(self.graph, graph_path)
        print(f"The lattice graph was saved at {graph_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature Lattice Graph Generation')
    parser.add_argument('--formula', type=str, default=LatticeGeneration.formula_idx, help='index of the formula')
    parser.add_argument('--config', type=str, default=LatticeGeneration.hyperparams_idx, help='index of configuration')
    parser.add_argument('--min_m', type=int, default=LatticeGeneration.min_m, help='min size of feature combinations')
    parser.add_argument('--max_m', type=int, default=LatticeGeneration.max_m, help='max size of feature combinations')
    parser.add_argument('--edge_sampling_ratio', type=float, default=LatticeGeneration.edge_sampling_ratio)
    parser.add_argument('--num_layers', type=int, default=GNN.num_layers)

    parser.add_argument('--within_level', type=bool, default=LatticeGeneration.within_level_edges,
                        help='add edges within the same level')
    # parser.add_argument('--hetero', type=bool, default=LatticeGeneration.is_hetero, help='create heterogeneous graph')
    parser.add_argument('--with_edge_attrs', type=bool, default=LatticeGeneration.with_edge_attrs,
                        help='add attributes to the edges')
    parser.add_argument('--data_name', type=str, default='loan',
                        help='name of dataset, options: {synthetic, loan, startup, mobile}')
    parser.add_argument('--print_tqdm', type=bool, default=True, help='whether to leave tqdm progress bars')
    args = parser.parse_args()

    if args.data_name == 'synthetic':
        dataset_path = f"GeneratedData/Formula{args.formula}/Config{args.config}/dataset.pkl"
    else:
        dataset_path = f"RealWorldData/{args.data_name}/dataset.pkl"

    start = time.time()
    lattice = FeatureLatticeGraph(dataset_path, args)
    end = time.time()
    print(f"Total time: {round(end - start, 4)} seconds")
    print(f"\n ============================================================\n")