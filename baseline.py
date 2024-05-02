import argparse
from GNN_models import LatticeGNN
from config import LatticeGeneration, GNN, Sampling, MissingDataConfig
from lattice_graph_generator_multiprocessing import FeatureLatticeGraph
from missing_data_masking import MissingDataMasking
from sampler import NodeSampler
from utils import *
from torch_geometric.nn import to_hetero
from sklearn.model_selection import train_test_split
import warnings
from Custom_KNN_Imputer import CustomKNNImputer
from sklearn.impute import SimpleImputer, KNNImputer

warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.metrics import mutual_info_score
import multiprocessing as mp
from functools import partial
import numpy as np
import torch
from itertools import combinations
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_pipeline_obj(args, seed, missing_indices_dict=None):
    # with open(f"{dir_path}missing_data_indices.pkl", 'rb') as f:
    #     missing_indices_dict = pickle.load(f)
    pipeline_obj = PipelineManager(args, seed, missing_indices_dict)
    return pipeline_obj


def get_dir_path(args):
    if args.data_name == 'synthetic':
        return f"GeneratedData/Formula{args.formula}/Config{args.config}/"
    else:
        return f"RealWorldData/{args.data_name}/"


class DataImputer:
    def __init__(self, missing_dict, imputation_method='mode'):
        self.missing_dict = missing_dict
        self.imputation_method = imputation_method

    def _replace_with_nan(self, data):
        for subgroup, features in self.missing_dict.items():
            g_id = int(subgroup.split('g')[-1])
            data.loc[data['subgroup'] == g_id, features] = np.nan
            # print(features)
            # print(data.loc[data['subgroup'] == g_id, features])
            # exit()
        return data

    def _impute_data(self, data):
        if self.imputation_method == 'mode':
            for column in data.columns:
                imputer = SimpleImputer(strategy='most_frequent')
                data[column] = imputer.fit_transform(data[[column]])
        elif self.imputation_method == 'KNN':
            imputer = CustomKNNImputer(n_neighbors=3)
            data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        elif self.imputation_method == 'constant':
            print("imputation is constant")
            imputer = SimpleImputer(strategy='constant', fill_value=1000)
            data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        else:
            raise ValueError("Unsupported imputation method")
        return data

    def process(self, data):
        data = self._replace_with_nan(data)
        data = self._impute_data(data)
        return data


class PipelineManager:
    def __init__(self, args, seed, missing_indices_dict=None):
        self.args = args
        self.seed = seed
        self.at_k = args.at_k if isinstance(args.at_k, list) else [args.at_k]
        self.graph_path, self.dir_path = read_paths(args)
        self.lattice_graph, self.subgroups = self._load_graph_information()
        self.feature_num = self.lattice_graph['g0'].x.shape[1]
        self.min_level = get_min_level(args.min_m, args.num_layers)
        self.max_level = get_max_level(args.max_m, args.num_layers, self.feature_num)
        self.restricted_graph_idxs_mapping = get_restricted_graph_idxs_mapping(self.feature_num, self.min_level,
                                                                               self.max_level)
        self.missing_indices_dict = self._get_missing_data_dict(missing_indices_dict)
        self.non_missing_dict = {subgroup: [idx for idx in range(self.lattice_graph[subgroup].num_nodes) if idx not in
                                            self.missing_indices_dict[subgroup]['all']] for subgroup in self.subgroups}
        self.train_idxs_dict, self.valid_idxs_dict = self._train_validation_split()
        self.test_idxs_dict = self._get_test_indices()

        df = pd.read_pickle(f"{self.dir_path}dataset.pkl")
        df = df.rename(str, axis='columns')
        df = self._impute(df)
        self.impute_mappings_dict = self._get_mi_after_impute(df)

        with open(f'{self.dir_path}dataset_{args.imputation_method}_missing={args.missing_prob}_seed{self.seed}.pkl',
                  'wb') as f:
            pickle.dump(self.impute_mappings_dict, f)

        self.preds = {subgroup: torch.zeros(self.lattice_graph[subgroup].num_nodes) for subgroup in self.subgroups}
        for gid in self.impute_mappings_dict:
            for comb_size in self.impute_mappings_dict[gid]:
                for comb in self.impute_mappings_dict[gid][comb_size]:
                    comb_idx = self.impute_mappings_dict[gid][comb_size][comb]['restricted_node_id']
                    comb_score = self.impute_mappings_dict[gid][comb_size][comb]['score']
                    self.preds[f"g{gid}"][comb_idx] = comb_score

    def _impute(self, df):
        missing_dict = {subgroup: list(self.missing_indices_dict[subgroup].keys())[:-1] for subgroup in
                        self.missing_indices_dict}
        imputer = DataImputer(missing_dict, self.args.imputation_method)
        print("Imputing missing data...")
        df = imputer.process(df)
        # df = df.astype(str)
        return df

    def _get_mi_after_impute(self, df):
        print("Computing MI scores for the imputed dataframe...")
        lattice_generator = FeatureLatticeGraph(f"{self.dir_path}dataset.pkl", self.args, df,
                                                create_edges=False, baseline=self.args.imputation_method)
        return lattice_generator.mappings_dict

    def _load_graph_information(self):
        lattice_graph = torch.load(self.graph_path)
        subgroups = lattice_graph.node_types
        return lattice_graph, subgroups

    def _get_missing_data_dict(self, missing_indices_dict):
        if missing_indices_dict is not None:
            return missing_indices_dict
        else:
            missing_indices_dict = MissingDataMasking(self.feature_num, self.subgroups, self.seed,
                                                      self.args.missing_prob, self.restricted_graph_idxs_mapping,
                                                      self.args.manual_md).missing_indices_dict
            with open(f"{self.dir_path}missing_data_indices_seed{self.seed}.pkl", 'wb') as f:
                pickle.dump(missing_indices_dict, f)
            return missing_indices_dict

    def _train_validation_split(self):
        sampler = NodeSampler(
            self.seed,
            self.min_level,
            self.max_level,
            self.feature_num,
            self.non_missing_dict,
            self.missing_indices_dict,
            self.restricted_graph_idxs_mapping,
            self.args.sampling_ratio,
            self.args.sampling_method,
        )
        train_idxs_dict = sampler.train_indices_dict
        valid_idxs_dict = sampler.val_indices_dict
        return train_idxs_dict, valid_idxs_dict

    def test_subgroup(self, subgroup, comb_size, show_results=True):
        gid = int(subgroup.split('g')[-1])
        test_indices = self.test_idxs_dict[subgroup]
        labels = self.lattice_graph[subgroup].y
        tmp_results_dict = compute_eval_metrics_baseline(labels, self.preds[subgroup], test_indices, self.at_k,
                                                         comb_size, self.feature_num)
        if show_results:
            print_results(tmp_results_dict, self.at_k, comb_size, subgroup)
        return tmp_results_dict

    def _get_test_indices(self):
        test_idxs_dict = dict()
        for subgroup in self.subgroups:
            test_idxs_dict[subgroup] = [idx for idx in range(self.lattice_graph[subgroup].num_nodes) if idx not in
                                        self.train_idxs_dict[subgroup] and idx not in self.valid_idxs_dict[subgroup]]
        return test_idxs_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds_num', type=int, default=3)
    parser.add_argument('--formula', type=str, default=str(LatticeGeneration.formula_idx))
    parser.add_argument('--config', type=str, default=str(LatticeGeneration.hyperparams_idx))
    parser.add_argument('--model', type=str, default=GNN.gnn_model)
    parser.add_argument('--hidden_channels', type=int, default=GNN.hidden_channels)
    parser.add_argument('--num_layers', type=int, default=GNN.num_layers)
    parser.add_argument('--p_dropout', type=float, default=GNN.p_dropout)
    parser.add_argument('--epochs', type=int, default=GNN.epochs)
    parser.add_argument('--sampling_ratio', type=float, default=Sampling.sampling_ratio)
    parser.add_argument('--sampling_method', type=str, default=Sampling.method)
    parser.add_argument('--valid_ratio', type=str, default=Sampling.validation_ratio)
    default_at_k = ','.join([str(i) for i in Evaluation.at_k])
    parser.add_argument('--at_k', type=lambda x: [int(i) for i in x.split(',')], default=default_at_k)
    parser.add_argument('--comb_size_list', type=int, default=Evaluation.comb_size_list)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--display', type=bool, default=False)
    parser.add_argument('--manual_md', type=bool, default=False, help='Manually input missing data')
    parser.add_argument('--min_m', type=int, default=LatticeGeneration.min_m, help='min size of feature combinations')
    parser.add_argument('--max_m', type=int, default=LatticeGeneration.max_m, help='max size of feature combinations')
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--data_name', type=str, default='loan', help='options:{synthetic, loan, startup, mobile}')
    parser.add_argument('--missing_prob', type=float, default=MissingDataConfig.missing_prob)
    parser.add_argument('--edge_sampling_ratio', type=float, default=LatticeGeneration.edge_sampling_ratio)
    parser.add_argument('--imputation_method', type=str, default='KNN', help="options: {KNN, mode}")
    parser.add_argument('--with_edge_attrs', type=bool, default=LatticeGeneration.with_edge_attrs,
                        help='add attributes to the edges')
    parser.add_argument('--print_tqdm', type=bool, default=True, help='whether to leave tqdm progress bars')
    parser.add_argument('--upward_analysis', type=bool, default=False, help='whether to leave tqdm progress bars')
    args = parser.parse_args()

    dir_path = get_dir_path(args)
    df = pd.read_pickle(dir_path + 'dataset.pkl')
    subgroups = [f'g{i}' for i in range(df['subgroup'].nunique())]
    del df
    results_dict = {comb_size: {seed: {subgroup: dict() for subgroup in subgroups}
                                for seed in range(1, args.seeds_num + 1)} for comb_size in args.comb_size_list}
    for seed in range(1, args.seeds_num + 1):
        set_seed(seed)
        with open(f"{dir_path}missing={args.missing_prob}_data_indices_seed{seed}.pkl", 'rb') as f:
            missing_indices_dict = pickle.load(f)
        pipeline_obj = get_pipeline_obj(args, seed, missing_indices_dict)
        subgroups = pipeline_obj.lattice_graph.x_dict.keys()
        print(f"Seed: {seed}\n=============================")
        for comb_size in args.comb_size_list:
            results_dict[comb_size][seed] = {g_id: pipeline_obj.test_subgroup(g_id, comb_size) for g_id in subgroups}
    save_results_baseline(results_dict, pipeline_obj.dir_path, args)