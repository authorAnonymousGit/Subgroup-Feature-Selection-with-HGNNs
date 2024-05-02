
import argparse
from GNN_models import LatticeGNN
from config import LatticeGeneration, GNN, Sampling, MissingDataConfig
from missing_data_masking import MissingDataMasking
from sampler import NodeSampler
from utils import *
import pandas as pd
from torch_geometric.nn import to_hetero
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def update_results_dict(results_dict, tmp_dict, eval_metrics, at_k, comb_size):
    for eval_metric in eval_metrics:
        for k in at_k:
            for subgroup in tmp_dict.keys():
                results_dict[eval_metric][k][comb_size] += tmp_dict[subgroup][eval_metric][k]
    return results_dict


def generate_df(dir_path, final_config_dict, eval_metrics, comb_size_list, at_k, args):
    data, index = [], []
    columns = [f"comb_size={comb_size}" for comb_size in comb_size_list]
    for eval_metric in eval_metrics:
        for k in at_k:
            index.append(f"{eval_metric}@{k}")
            data.append([final_config_dict[eval_metric][k][comb_size] for comb_size in comb_size_list])
    df = pd.DataFrame(data, index=index, columns=columns)
    if args.imputation_method is None:
        df.to_csv(f"{dir_path}Model={args.model}_Config{config_idx}_SamplingRatio={args.sampling_ratio}_"
                  f"MissingRatio={args.missing_prob}_SamplingMethod={args.sampling_method}_lr={args.lr}_"
                  f"EdgeSamplingRatio={args.edge_sampling_ratio}_hidden_channels={args.hidden_channels}_"
                  f"num_layers={args.num_layers}_.csv")
    else:
        df.to_csv(f"{dir_path}Config{config_idx}_MissingRatio={args.missing_prob}_"
                  f"imputation_method={args.imputation_method}.csv")
    return


def get_formula_and_config_lists(data_name):
    if data_name == 'synthetic':
        formula_idx_list = Evaluation.formula_idx_list
        config_idx_list = Evaluation.config_idx_list
    else:
        formula_idx_list = [1]
        config_idx_list = [1]
    return formula_idx_list, config_idx_list


def get_dir_path(data_name):
    if data_name == 'synthetic':
        return 'GeneratedData/'
    else:
        return f'RealWorldData/{args.data_name}/'


def get_curr_dir_path(args, formula_idx, config_idx):
    if args.data_name == 'synthetic':
        return dir_path + f"Formula{formula_idx}/Config{config_idx}/"
    else:
        return dir_path


def get_pkl_path(args, curr_dir_path, comb_size):
    if args.imputation_method is None:
        return (f"{curr_dir_path}combSize={comb_size}_"
                f"samplingRatio={args.sampling_ratio}_missingRatio={args.missing_prob}_"
                f"samplingMethod={args.sampling_method}_edgeSamplingRatio={args.edge_sampling_ratio}_"
                f"gamma={args.gamma}_lr={args.lr}_hidden_channels={args.hidden_channels}_num_layers={args.num_layers}"
                f"_model={args.model}.pkl")
    else:
        return (f"{curr_dir_path}combSize={comb_size}_"
                f"missingRatio={args.missing_prob}_imputation_method={args.imputation_method}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds_num', type=int, default=3)
    parser.add_argument('--missing_prob', type=str, default=str(MissingDataConfig.missing_prob))
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--sampling_ratio', type=str, default=str(Sampling.sampling_ratio))
    parser.add_argument('--sampling_method', type=str, default=Sampling.method)
    default_at_k = ','.join([str(i) for i in Evaluation.at_k])
    parser.add_argument('--at_k', type=lambda x: [int(i) for i in x.split(',')], default=default_at_k)
    parser.add_argument('--comb_size_list', type=int, default=Evaluation.comb_size_list)
    parser.add_argument('--edge_sampling_ratio', type=float, default=LatticeGeneration.edge_sampling_ratio)
    # parser.add_argument('--dir_path', type=str, default='GeneratedData/', help='Path to the results directory')
    parser.add_argument('--data_name', type=str, default='synthetic',
                        help='name of dataset, options: {synthetic, loan, startup, mobile}')
    parser.add_argument('--model', type=str, default=GNN.gnn_model, help='Path to the results directory')
    parser.add_argument('--imputation_method', type=str, default=None, help="options: {KNN, mode}")
    parser.add_argument('--hidden_channels', type=int, default=GNN.hidden_channels)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=1)
    args = parser.parse_args()

    formula_idx_list, config_idx_list = get_formula_and_config_lists(args.data_name)
    eval_metrics = Evaluation.eval_metrics
    comb_size_list = args.comb_size_list
    at_k = args.at_k if isinstance(args.at_k, list) else [args.at_k]
    dir_path = get_dir_path(args.data_name)
    for config_idx in config_idx_list:
        config_results_dict = {eval_metric: {k: {comb_size: 0.0 for comb_size in comb_size_list} for k in at_k}
                               for eval_metric in eval_metrics}
        for formula_idx in formula_idx_list:
            for comb_size in args.comb_size_list:
                curr_dir_path = get_curr_dir_path(args, formula_idx, config_idx)
                pkl_path = get_pkl_path(args, curr_dir_path, comb_size)
                tmp_dict = pd.read_pickle(pkl_path)
                config_results_dict = update_results_dict(config_results_dict, tmp_dict, eval_metrics, at_k, comb_size)
        subgroup_num = len(tmp_dict.keys())
        final_config_dict = {eval_metric: {k: {comb_size: round(config_results_dict[eval_metric][k][comb_size] /
                                                                (subgroup_num*len(formula_idx_list)), 3)
                                               for comb_size in comb_size_list}
                                           for k in at_k} for eval_metric in eval_metrics}
        generate_df(dir_path, final_config_dict, eval_metrics, comb_size_list, at_k, args)

