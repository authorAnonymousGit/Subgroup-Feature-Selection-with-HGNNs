
import argparse
from GNN_models import LatticeGNN
from config import LatticeGeneration, GNN, Sampling, MissingDataConfig
from missing_data_masking import MissingDataMasking
from sampler import NodeSampler
from utils import *
from torch_geometric.nn import to_hetero
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_pipeline_obj(args, seed):
    pipeline_obj = PipelineManager(args, seed)
    return pipeline_obj


class PipelineManager:
    def __init__(self, args, seed, missing_indices_dict=None):
        self.args = args
        self.seed = seed
        self.config_idx = int(args.config)
        self.epochs = args.epochs
        self.at_k = args.at_k if isinstance(args.at_k, list) else [args.at_k]
        self.graph_path, self.dir_path = read_paths(args)
        self.lattice_graph, self.subgroups = self._load_graph_information()
        self.feature_num = self.lattice_graph['g0'].x.shape[1]
        self.min_level = get_min_level(args.min_m, args.num_layers)
        self.max_level = get_max_level(args.max_m, args.num_layers, self.feature_num)
        self.restricted_graph_idxs_mapping = get_restricted_graph_idxs_mapping(self.feature_num, self.min_level,
                                                                               self.max_level)
        self.missing_indices_dict = self._get_missing_data_dict(missing_indices_dict)
        self.non_missing_dict = self._get_non_missing_dict()
        self.train_idxs_dict, self.valid_idxs_dict = self._train_validation_split()
        self.test_idxs_dict = self._get_test_indices()

    def _load_graph_information(self):
        lattice_graph = torch.load(self.graph_path)
        subgroups = lattice_graph.node_types
        return lattice_graph, subgroups

    def _get_non_missing_dict(self):
        non_missing_dict_sets = {subgroup: set(self.missing_indices_dict[subgroup]['all'])
                                 for subgroup in self.subgroups}
        return {subgroup: [idx for idx in range(self.lattice_graph[subgroup].num_nodes) if idx not in
                           non_missing_dict_sets[subgroup]] for subgroup in self.subgroups}

    def _get_missing_data_dict(self, missing_indices_dict):
        if missing_indices_dict is not None:
            return missing_indices_dict
        else:
            missing_indices_dict = MissingDataMasking(self.feature_num, self.subgroups, self.seed,
                                                      self.args.missing_prob, self.restricted_graph_idxs_mapping,
                                                      self.args.manual_md).missing_indices_dict
            with open(f"{self.dir_path}missing={self.args.missing_prob}_data_indices_seed{self.seed}.pkl", 'wb') as f:
                pickle.dump(missing_indices_dict, f)
            return missing_indices_dict

    def _init_model_optim(self):
        model = LatticeGNN(self.args.model, self.feature_num, self.args.hidden_channels, self.args.num_layers,
                           self.args.p_dropout)
        model = to_hetero(model, self.lattice_graph.metadata())
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        return model, optimizer

    def test_subgroup(self, subgroup, comb_size, show_results=True):
        test_indices = self.test_idxs_dict[subgroup]
        self.lattice_graph.to(device)
        model = torch.load(f"{self.dir_path}{self.args.model}_seed{seed}_{subgroup}.pt")
        model.to(device)
        model.eval()
        with torch.no_grad():
            out = model(self.lattice_graph.x_dict, self.lattice_graph.edge_index_dict)
        labels = self.lattice_graph[subgroup].y
        preds = out[subgroup]
        tmp_results_dict = compute_eval_metrics(labels, preds, test_indices, self.at_k, comb_size, self.feature_num)
        if show_results:
            print_results(tmp_results_dict, self.at_k, comb_size, subgroup)
        return tmp_results_dict

    def _train_validation_split(self):
        sampler = NodeSampler(
            self.config_idx,
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

    def _get_test_indices(self):
        test_idxs_dict = dict()
        for subgroup in self.subgroups:
            test_idxs_dict[subgroup] = [idx for idx in range(self.lattice_graph[subgroup].num_nodes) if idx not in
                                        self.train_idxs_dict[subgroup] and idx not in self.valid_idxs_dict[subgroup]]
        return test_idxs_dict

    def _run_training_epoch(self, train_indices, model, subgroup, optimizer, criterion):
        model.train()
        optimizer.zero_grad()
        out = model(self.lattice_graph.x_dict, self.lattice_graph.edge_index_dict)
        labels = self.lattice_graph[subgroup].y[train_indices]
        predictions = out[subgroup][train_indices]
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    def _get_validation_loss(self, validation_indices, model, subgroup, criterion):
        model.eval()
        model.to(device)
        with torch.no_grad():
            out = model(self.lattice_graph.x_dict, self.lattice_graph.edge_index_dict)
        labels = self.lattice_graph[subgroup].y[validation_indices]
        predictions = out[subgroup][validation_indices]
        loss = criterion(predictions, labels)
        return loss.item()

    def _run_over_validation(self, validation_indices, model, subgroup, criterion, best_val, no_impr_counter, seed):
        loss_validation = self._get_validation_loss(validation_indices, model, subgroup, criterion)
        if loss_validation < best_val:
            best_val = loss_validation
            no_impr_counter = 0
            torch.save(model, f"{self.dir_path}{self.args.model}_seed{seed}_{subgroup}.pt")
        else:
            no_impr_counter += 1
        return best_val, no_impr_counter

    def train_model(self, seed):
        torch.manual_seed(seed)
        criterion = torch.nn.MSELoss()
        self.lattice_graph.to(device)
        for subgroup in self.subgroups:
            print(f"\nTraining on subgroup {subgroup}...")
            model, optimizer = self._init_model_optim()
            train_indices, validation_indices = self.train_idxs_dict[subgroup], self.valid_idxs_dict[subgroup]
            no_impr_counter = 0
            epochs_stable_val = GNN.epochs_stable_val
            best_val = float('inf')
            for epoch in range(1, self.epochs + 1):
                if no_impr_counter == epochs_stable_val:
                    break
                loss_value = self._run_training_epoch(train_indices, model, subgroup, optimizer, criterion)
                if epoch == 1 or epoch % 5 == 0:
                    print(f'Epoch: {epoch}, Train Loss: {round(loss_value, 4)}, Best Val: {round(best_val, 4)}')
                    if not self.args.save_model:
                        continue
                    best_val, no_impr_counter = self._run_over_validation(validation_indices, model, subgroup,
                                                                          criterion, best_val, no_impr_counter, seed)


    def model_not_found(self, seed):
        for subgroup in self.subgroups:
            path = f"{self.dir_path}{self.args.model}_seed{seed}_{subgroup}.pt"
            if not os.path.exists(path):
                return True
        return False


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
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--display', type=bool, default=False)
    parser.add_argument('--manual_md', type=bool, default=False, help='Manually input missing data')
    parser.add_argument('--min_m', type=int, default=LatticeGeneration.min_m, help='min size of feature combinations')
    parser.add_argument('--max_m', type=int, default=LatticeGeneration.max_m, help='max size of feature combinations')
    parser.add_argument('--edge_sampling_ratio', type=float, default=LatticeGeneration.edge_sampling_ratio)
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--data_name', type=str, default='loan', help='options:{synthetic, loan, startup, mobile}')
    parser.add_argument('--missing_prob', type=float, default=MissingDataConfig.missing_prob)
    args = parser.parse_args()

    dir_path = get_dir_path(args)
    pipeline_obj = get_pipeline_obj(args, 0)
    subgroups = pipeline_obj.lattice_graph.x_dict.keys()
    results_dict = {comb_size: {seed: {subgroup: dict() for subgroup in subgroups}
                                for seed in range(1, args.seeds_num + 1)} for comb_size in args.comb_size_list}

    for seed in range(1, args.seeds_num + 1):
        set_seed(seed)
        pipeline_obj = get_pipeline_obj(args, seed)
        print(f"Seed: {seed}\n=============================")
        pipeline_obj.train_model(seed)
        for comb_size in args.comb_size_list:
            results_dict[comb_size][seed] = {g_id: pipeline_obj.test_subgroup(g_id, comb_size) for g_id in subgroups}
    save_results(results_dict, pipeline_obj.dir_path, args)

