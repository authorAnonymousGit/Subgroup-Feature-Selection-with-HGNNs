
import argparse
from config import MLP
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Linear
import torch.nn.functional as F
from utils import *
from train_and_evaluate import *
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, p_dropout):
        super(MLPModel, self).__init__()
        self.p_dropout = p_dropout
        self.num_layers = num_layers
        self._set_layers(input_size, hidden_size)
        self.out = Linear(hidden_size, 1)

    def _set_layers(self, input_size, hidden_size):
        for layer_idx in range(1, self.num_layers + 1):
            if layer_idx == 1:
                setattr(self, f'fc{layer_idx}', Linear(input_size, hidden_size))
            else:
                setattr(self, f'fc{layer_idx}', Linear(hidden_size, hidden_size))
        return

    def forward(self, x):
        for layer_idx in range(1, self.num_layers + 1):
            fc = getattr(self, f'fc{layer_idx}')
            x = F.leaky_relu(fc(x))
            x = F.dropout(x, p=self.p_dropout, training=self.training)
        output = self.out(x).squeeze()
        return output


def get_input_vectors(indices, feature_num):
    x_input = []
    for node_id in indices:
        binary_vec = convert_decimal_to_binary(node_id + 1, feature_num)
        binary_vec = [int(digit) for digit in binary_vec]
        x_input.append(binary_vec)
    return torch.tensor(x_input, dtype=torch.float32).to(device)



def train_mlp_model(pipeline_obj, subgroups, args):
    torch.manual_seed(seed)
    criterion = torch.nn.MSELoss()
    lattice_graph = pipeline_obj.lattice_graph
    lattice_graph.to(device)
    dir_path = pipeline_obj.dir_path
    feature_num = pipeline_obj.feature_num
    for subgroup in subgroups:
        print(f"\nTraining on subgroup {subgroup}...")
        model = MLPModel(feature_num, args.hidden_channels, args.num_layers, args.p_dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        train_indices = pipeline_obj.train_idxs_dict[subgroup]
        valid_indices = pipeline_obj.valid_idxs_dict[subgroup]
        x_train = get_input_vectors(train_indices, feature_num)
        x_valid = get_input_vectors(valid_indices, feature_num)
        no_impr_counter = 0
        epochs_stable_val = MLP.epochs_stable_val
        best_val = float('inf')
        for epoch in range(args.epochs):
            if no_impr_counter == epochs_stable_val:
                break
            loss_value = run_training_epoch(train_indices, x_train, model, subgroup, optimizer, criterion, lattice_graph)
            if epoch == 1 or epoch % 5 == 0:
                print(f'Epoch: {epoch}, Train Loss: {round(loss_value, 4)}, Best Val: {round(best_val, 4)}')
                if not args.save_model:
                    continue
                best_val, no_impr_counter = run_over_validation(lattice_graph, valid_indices, x_valid,
                                                                model, subgroup, criterion, best_val, no_impr_counter,
                                                                seed, dir_path)
    return


def run_training_epoch(train_indices, x_train, model, subgroup, optimizer, criterion, lattice_graph):
    model.train()
    optimizer.zero_grad()
    preds = model(x_train)
    labels = lattice_graph[subgroup].y[train_indices]
    loss = criterion(preds, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def run_over_validation(lattice_graph, validation_indices, x_valid, model, subgroup, criterion,
                        best_val, no_impr_counter, seed, dir_path):
    loss_validation = get_validation_loss(lattice_graph, validation_indices, x_valid, model, subgroup, criterion)
    if loss_validation < best_val:
        best_val = loss_validation
        no_impr_counter = 0
        torch.save(model, f"{dir_path}{'MLP_model'}_seed{seed}_{subgroup}.pt")
    else:
        no_impr_counter += 1
    return best_val, no_impr_counter


def get_validation_loss(lattice_graph, validation_indices, x_valid, model, subgroup, criterion):
    model.eval()
    model.to(device)
    with torch.no_grad():
        preds = model(x_valid)
    labels = lattice_graph[subgroup].y[validation_indices]
    loss = criterion(preds, labels)
    return loss.item()


def test_subgroup(pipeline_obj, subgroup, comb_size, show_results=True):
    test_indices = pipeline_obj.test_idxs_dict[subgroup]
    x_test = get_input_vectors(test_indices, pipeline_obj.feature_num)
    lattice_graph = pipeline_obj.lattice_graph
    lattice_graph.to(device)
    model = torch.load(f"{pipeline_obj.dir_path}{'MLP_model'}_seed{seed}_{subgroup}.pt")
    model.to(device)
    model.eval()
    with torch.no_grad():
        preds = model(x_test)
    labels = lattice_graph[subgroup].y
    tmp_results_dict = compute_eval_metrics(labels, preds, test_indices, pipeline_obj.at_k,
                                            comb_size, pipeline_obj.feature_num)
    if show_results:
        print_results(tmp_results_dict, pipeline_obj.at_k, comb_size, subgroup)
    return tmp_results_dict


def save_results_MLPModel(test_results, dir_path, comb_size_list, args):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    for comb_size in comb_size_list:
        results_path = dir_path + (f'combSize={comb_size}_samplingRatio={args.sampling_ratio}_'
                                   f'missingRatio={args.missing_prob}_samplingMethod={args.sampling_method}_'
                                   f'edgeSamplingRatio={args.edge_sampling_ratio}_gamma={args.gamma}_'
                                   f'lr={args.lr}_model=MLP.pkl')
        final_test_results = comp_ave_results(test_results[comb_size])
        with open(results_path, 'wb') as f:
            pickle.dump(final_test_results, f)
    save_hyperparams_MLP(dir_path, args)
    return


def save_hyperparams_MLP(dir_path, args):
    hyperparams = (f"Hidden channels: {args.hidden_channels}\n"
                   f"Number of layers: {args.num_layers}\nDropout: {args.p_dropout}\nlr: {args.lr}\n"
                   f"weight_decay: {args.weight_decay}\n")
    hyperparams_path = dir_path + 'hyperparams_MLP.txt'
    with open(hyperparams_path, 'w') as f:
        f.write(hyperparams)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds_num', type=int, default=3)
    parser.add_argument('--formula', type=str, default=str(LatticeGeneration.formula_idx))
    parser.add_argument('--config', type=str, default=str(LatticeGeneration.hyperparams_idx))
    parser.add_argument('--hidden_channels', type=int, default=MLP.hidden_channels)
    parser.add_argument('--num_layers', type=int, default=MLP.num_layers)
    parser.add_argument('--p_dropout', type=float, default=MLP.p_dropout)
    parser.add_argument('--epochs', type=int, default=MLP.epochs)
    parser.add_argument('--missing_prob', type=float, default=MissingDataConfig.missing_prob)
    parser.add_argument('--sampling_ratio', type=float, default=Sampling.sampling_ratio)
    parser.add_argument('--sampling_method', type=str, default=Sampling.method)
    parser.add_argument('--valid_ratio', type=str, default=Sampling.validation_ratio)
    default_at_k = ','.join([str(i) for i in Evaluation.at_k])
    parser.add_argument('--at_k', type=lambda x: [int(i) for i in x.split(',')], default=default_at_k)
    parser.add_argument('--comb_size_list', type=int, default=Evaluation.comb_size_list)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--display', type=bool, default=False)
    parser.add_argument('--min_m', type=int, default=LatticeGeneration.min_m, help='min size of feature combinations')
    parser.add_argument('--max_m', type=int, default=LatticeGeneration.max_m, help='max size of feature combinations')
    parser.add_argument('--manual_md', type=bool, default=False, help='Manually input missing data')
    parser.add_argument('--load_model', type=bool, default=True, help='Used for loading the pipeline manager object')
    parser.add_argument('--load_mlp_model', type=bool, default=False)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--data_name', type=str, default='synthetic', help='options:{synthetic, loan, startup, mobile}')
    parser.add_argument('--dir_path', type=str, default=None)
    parser.add_argument('--edge_sampling_ratio', type=float, default=LatticeGeneration.edge_sampling_ratio)

    args = parser.parse_args()
    seeds_num = args.seeds_num
    dir_path = get_dir_path(args)
    pipeline_obj = get_pipeline_obj(args, 0)
    subgroups = pipeline_obj.lattice_graph.x_dict.keys()
    results_dict = {comb_size: {seed: {subgroup: dict() for subgroup in subgroups}
                                for seed in range(1, seeds_num + 1)} for comb_size in args.comb_size_list}

    for seed in range(1, seeds_num + 1):
        set_seed(seed)
        pipeline_obj = get_pipeline_obj(args, seed)
        if not args.load_mlp_model or pipeline_obj.model_not_found(seed):
            print(f"Seed: {seed}\n=============================")
            train_mlp_model(pipeline_obj, subgroups, args)
        for comb_size in args.comb_size_list:
            results_dict[comb_size][seed] = {g_id: test_subgroup(pipeline_obj, g_id, comb_size) for g_id in subgroups}
    save_results_MLPModel(results_dict, pipeline_obj.dir_path, args.comb_size_list, args)

