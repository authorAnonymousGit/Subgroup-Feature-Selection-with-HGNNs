import json
import pickle
import numpy as np
import os
import pandas as pd
from config import LatticeGeneration


class LogicalDatasetGenerator:
    def __init__(self, formula_name, config_num, formula, hyperparams):
        np.random.seed(int(config_num))
        self.seed = int(config_num)
        self.name = formula_name
        self.subgroups_num = hyperparams['subgroups_num']
        self.feature_num = hyperparams['feature_num']
        self.feature_range = hyperparams['feature_range']  # requirement - list from 0 to n with increments of 1
        self.formula = formula
        self.hyperparams = hyperparams
        self.feature_pool = [f'f_{i}' for i in range(self.feature_num)]
        self.operator_dict = self._get_operator_dict()
        self.relevant_features = self._get_relevant_features()
        self.redundant_features = self._get_redundant_features()
        self.correlated_features = self._get_correlated_features()
        self.noisy_features = self.feature_pool
        self.dataset = self.generate_values()

    @staticmethod
    def _get_operator_dict():
        op_dict = {'and': np.bitwise_and, 'or': np.bitwise_or, 'xor': np.bitwise_xor, '+': np.add, 
                   '-': np.subtract, '*': np.multiply, '^': np.power}
        return op_dict

    def _get_relevant_features(self):
        """
        Extracts the relevant features from the formula and removes them from the feature pool.
        """
        relevant_features = [expr for expr in self.formula.split(' ') if expr not in self.operator_dict.keys() 
                             and expr not in ['(', ')']]
        self.feature_pool = [f for f in self.feature_pool if f not in relevant_features]
        return relevant_features

    def _get_redundant_features(self):
        """
        Randomly selects a number of features to be redundant and generates a formula for each of them,
        removing them from the feature pool.
        """
        redundant_features = list(np.random.choice(self.feature_pool, self.hyperparams['redundant_num'], replace=False))
        self.feature_pool = [f for f in self.feature_pool if f not in redundant_features]
        redundant_dict = self._generate_redundant_expression(redundant_features)
        return redundant_dict

    def _generate_redundant_expression(self, redundant_features):
        """
        Generates a random expression for a redundant feature, containing some combination of relevant features.
        """
        min_subset_size = self.hyperparams['redundant_min_subgroup']
        max_subset_size = self.hyperparams['redundant_max_subgroup']
        expressions = {}
        operation_list = list(self.operator_dict.keys())
        for f in redundant_features:
            subset_size = np.random.randint(min_subset_size, max_subset_size + 1)
            subset = np.random.choice(self.relevant_features, subset_size)
            operations = np.random.choice(operation_list, subset_size - 1)
            interlacing_string = ' '.join([f'{subset[i]} {operations[i]}'
                                           for i in range(subset_size - 1)]) + f' {subset[-1]}'
            expressions[f] = interlacing_string
        return expressions

    def _get_correlated_features(self):
        """
        Randomly selects a number of features to be correlated with the label and generates a probability for each
        feature to be equal to the label, removing them from the feature pool.
        """
        correlated_features = list(np.random.choice(self.feature_pool,
                                   self.hyperparams['correlated_num'], replace=False))
        self.feature_pool = [f for f in self.feature_pool if f not in correlated_features]
        min_prob = self.hyperparams['correlated_min_probability']
        max_prob = self.hyperparams['correlated_max_probability']
        correlated_dict = {f: np.random.uniform(min_prob, max_prob) for f in correlated_features}
        return correlated_dict

    def generate_values(self):
        """
        Creates the dataframe by generating random values for the features and computing the label according to the
        formula. Then, it adds the remaining values according to the hyperparameters.
        """
        sample_size = self.hyperparams['sample_size']
        tmp_dataset = dict()
        for feature in self.relevant_features:
            tmp_dataset[feature] = np.random.choice(self.feature_range, sample_size)
        tmp_dataset['y'] = self._compute_by_formula(tmp_dataset, self.formula)
        tmp_dataset = self._modify_by_features(tmp_dataset, sample_size)
        df = self._generate_dataframe(tmp_dataset)
        return df

    def _modify_by_features(self, tmp_dataset, sample_size):
        tmp_dataset = self._get_correlated_vals(tmp_dataset, sample_size)
        tmp_dataset = self._get_redundant_vals(tmp_dataset, sample_size)
        tmp_dataset = self._get_noisy_vals(tmp_dataset, sample_size)
        return tmp_dataset

    def _generate_dataframe(self, tmp_dataset):
        df = pd.DataFrame(tmp_dataset)
        df = df[[f'f_{i}' for i in range(self.feature_num)] + ['y']]
        df['subgroup'] = np.random.randint(0, self.subgroups_num, len(df))
        df = self._add_random_noise(df)
        return df

    def infix_to_postfix(self, expression):
        precedence = {'and': 3, 'or': 2, 'xor': 4, '+': 1, '-': 1, '*': 2, '^': 5}
        output = []
        operators = []
        formula = expression.split(' ')
        for token in formula:
            if token not in self.operator_dict.keys() and token not in '()':
                output.append(token)
            elif token == '(':
                operators.append(token)
            elif token == ')':
                while operators[-1] != '(':
                    output.append(operators.pop())
                operators.pop()
            else:
                while operators and operators[-1] != '(' and precedence[operators[-1]] >= precedence[token]:
                    output.append(operators.pop())
                operators.append(token)
        while operators:
            output.append(operators.pop())
        return output

    def _compute_by_formula(self, dataset, formula):
        reverse_polish = self.infix_to_postfix(formula)
        clause_results = []
        for clause in reverse_polish:
            if clause not in self.operator_dict.keys():
                clause_results.append(dataset[clause])
            else:
                op_func = self.operator_dict[clause]
                clause2_result = clause_results.pop()
                clause1_result = clause_results.pop()
                if clause in '+-*^':
                    result = np.mod(op_func(clause1_result, clause2_result), len(self.feature_range))
                else:
                    result = op_func(clause1_result, clause2_result)
                clause_results.append(result)
        return clause_results[0].astype(int)

    def _get_correlated_vals(self, dataset, sample_size):
        for feature in self.correlated_features:
            dataset[feature] = np.where(np.random.rand(sample_size) <
                                        self.correlated_features[feature],
                                        dataset['y'], np.random.choice(self.feature_range))
        return dataset

    def _get_redundant_vals(self, dataset, sample_size):
        redundant_flip_prob = self.hyperparams['redundant_flip_probability']
        for feature in self.redundant_features:
            computed_vals = self._compute_by_formula(dataset, self.redundant_features[feature])
            dataset[feature] = np.where(np.random.rand(sample_size) > redundant_flip_prob,
                                        computed_vals, np.random.choice(self.feature_range))
        return dataset

    def _get_noisy_vals(self, dataset, sample_size):
        for feature in self.noisy_features:
            dataset[feature] = np.random.choice(self.feature_range, sample_size)
        return dataset

    def _add_random_noise(self, df):
        random_noise_mean = self.hyperparams['random_noise_mean']
        random_noise_std = self.hyperparams['random_noise_std']
        if random_noise_mean < 0:
            # Without random noise
            return df
        feature_list = [f'f_{i}' for i in range(self.feature_num)]
        for subgroup in range(self.subgroups_num):
            tmp_df = df[df['subgroup'] == subgroup].copy()
            subgroup_noise = np.random.normal(random_noise_mean, random_noise_std)
            for feature in feature_list:
                noise_flip_prob = np.random.normal(subgroup_noise, subgroup_noise/4)
                tmp_df[feature] = tmp_df[feature].apply(lambda x: x if np.random.rand() < noise_flip_prob
                                                        else np.random.choice(self.feature_range)).astype(int)
            df.loc[tmp_df.index] = tmp_df
        return df

    def create_description(self):
        description = """"""
        description += f"Formula {self.name}: {self.formula}\n"
        description += f"Seed: {self.seed}\n"
        description = self._add_hyperparams_to_description(description)
        description += f"Relevant features: {', '.join(sorted(list(self.relevant_features)))}\n\n"
        description += f"Redundant features and the formulas that define them: \n\n"
        for feature, formula in self.redundant_features.items():
            description += f"\t{feature}: {formula}\n"
        description += f"\nCorrelated features and probabilities of being equal to label: \n"
        for feature, prob in self.correlated_features.items():
            description += f"\t{feature}: {round(prob, 3)}\n"
        return description

    def _add_hyperparams_to_description(self, description):
        description += "========================================\n"
        description += "Hyperparameters:\n"
        for key, value in self.hyperparams.items():
            description += f"\t{key}: {value}\n"
        description += "========================================\n"
        description += "\n"
        return description

    def save(self):
        if not self.save:
            return
        path = f'GeneratedData/Formula{self.name}/Config{self.seed}'
        if not os.path.exists(path):
            os.makedirs(path)
        pickle.dump(self.dataset, open(f'{path}/dataset.pkl', 'wb'))
        with open(f'{path}/description.txt', 'w') as f:
            f.write(self.create_description())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate synthetic data for logical formulas')
    parser.add_argument('--formula', type=str, default=LatticeGeneration.formula_idx, help='Index of the formula to be generated')
    parser.add_argument('--config', type=str, default=LatticeGeneration.hyperparams_idx, help='Index of the configuration to be generated')
    args = parser.parse_args()

    configs = json.load(open('data_generation_config.json'))
    data = LogicalDatasetGenerator(args.formula, args.config, configs['formulas'][str(args.formula)]['formula'],
                                   configs['hyperparams'][str(args.config)])
    data.save()

