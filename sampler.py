import random
import numpy as np
from numpy import math
import typing as t
from config import Sampling
from sklearn.model_selection import train_test_split


class NodeSampler:
    def __init__(self, config_idx, min_level, max_level, feature_num, non_missing_dict, missing_indices_dict,
                 restricted_graph_idxs_mapping, sampling_ratio, sampling_method):
        # np.random.seed(config_idx)
        self.min_level = min_level
        self.max_level = max_level
        self.subgroups = non_missing_dict.keys()
        self.feature_num = feature_num
        self.non_missing_dict = non_missing_dict
        self.missing_indices_dict = missing_indices_dict
        self.restricted_graph_idxs_mapping = restricted_graph_idxs_mapping
        self.sampling_ratio = sampling_ratio
        self.sampling_method = sampling_method
        self.validation_ratio = Sampling.validation_ratio
        self.sampling_func = self._get_sampling_func_dict()
        self.train_indices_dict, self.val_indices_dict = self._get_samples()

    def _get_sampling_func_dict(self):
        sampling_funcs = {
            'arbitrary': self._arbitrary_sampling,
            'randwalk': self._uniform_sampling,
        }
        if self.sampling_method not in sampling_funcs:
            raise ValueError(f"Invalid sampling method: {self.sampling_method}")
        return sampling_funcs[self.sampling_method]

    def _get_samples(self):
        if self.sampling_ratio < 1:
            sampled_indices_dict = self.sampling_func()
        else:
            sampled_indices_dict = self.non_missing_dict
        train_indices_dict, val_indices_dict = dict(), dict()
        for gid, indices in sampled_indices_dict.items():
            train_indices, val_indices = train_test_split(indices, test_size=self.validation_ratio)
            train_indices_dict[gid] = train_indices
            val_indices_dict[gid] = val_indices
        return train_indices_dict, val_indices_dict

    def _arbitrary_sampling(self):
        nids = dict()
        for gid in self.subgroups:
            num_samples = int(self.sampling_ratio * len(self.non_missing_dict[gid]))
            nids[gid] = list(np.random.choice(self.non_missing_dict[gid], num_samples, replace=False))
        return nids

    def _random_walk(self, curr_node: str, node_list: t.List[str], present_bits: t.List[int]) -> str:
        rand_idx = self.feature_num - 1 - np.random.choice(present_bits)
        rand_bit = np.random.choice(['0', '1'], p=[0.5, 0.5])
        new_node = curr_node[: rand_idx] + rand_bit + curr_node[rand_idx + 1:]
        if new_node not in node_list:
            curr_node = new_node
            if self.min_level <= new_node.count('1') <= self.max_level:
                node_list.append(new_node)
        return curr_node

    def _get_num_samples(self, non_missing_fids):
        non_missing_num = len(non_missing_fids)
        maximal_num_samples = sum([math.comb(non_missing_num, ell) for ell in range(self.min_level, self.max_level+1)])
        return int(self.sampling_ratio * maximal_num_samples)

    def _get_start_node(self, present_bits):
        start_node = [0] * self.feature_num
        while True:
            random_features = np.random.choice([0, 1], size=len(present_bits), p=[0.5, 0.5])
            if self.min_level <= sum(random_features) <= self.max_level:
                break
        for i, bit in enumerate(random_features):
            start_node[self.feature_num - 1 - present_bits[i]] = bit
        return ''.join([str(bit) for bit in start_node])

    def _uniform_sampling(self):
        sampled_nids_dict = dict()
        for subgroup in self.subgroups:
            missing_fids = [int(feat.split('_')[-1]) for feat in self.missing_indices_dict[subgroup].keys() if
                            'f_' in feat]
            non_missing_fids = sorted(list(set(range(self.feature_num)) - set(missing_fids)))
            num_samples = self._get_num_samples(non_missing_fids)
            start_node = self._get_start_node(non_missing_fids)
            node_list = [start_node]
            curr_node = start_node
            stuck_rounds = 0
            while len(node_list) < num_samples:
                len_before_walk = len(node_list)
                curr_node = self._random_walk(curr_node, node_list, non_missing_fids)
                len_after_walk = len(node_list)
                if len_after_walk > len_before_walk:
                    stuck_rounds = 0
                else:
                    stuck_rounds += 1
                if stuck_rounds == 50:  # explore from the beginning
                    curr_node = self._restart_walk(non_missing_fids, node_list)
                    stuck_rounds = 0
            orig_sampled_nids = list(map(lambda bstr: int(bstr, 2) - 1, node_list))
            sampled_nids_dict[subgroup] = [self.restricted_graph_idxs_mapping[oid] for oid in orig_sampled_nids]
        return sampled_nids_dict

    def _restart_walk(self, present_bits, node_list):
        start_node = self._get_start_node(present_bits)
        while start_node in node_list:
            start_node = self._get_start_node(present_bits)
        return start_node
