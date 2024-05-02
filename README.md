# Subgroup-Feature-Selection-with-HGNNs

More detailed instructions and explanations will be uploaded soon.
********************************************************************************
# Experimental Instructions
**1)** Generate the multiple lattice graph for each dataset:

```bash generate_multiple_lattice_graphs.sh <dataset_name1> <dataset_name2> ...```

where dataset_name1, dataset_name2, ... are the names of the datasets.
The script currently supports the following options: {"synthetic", "loan", "startup", "mobile"}.
Note that for the synthetic dataset, the formula and configs indexes are determined inside the script.\
The list of edge_sampling_ratio is also defined in the script.

**2)** Train and evaluate MISFEAT:

```bash run_experiments.sh <dataset_name1> <dataset_name2> ...```

where dataset_name1, dataset_name2, ... are the names of the datasets.
The script supports the following options: {"synthetic", "loan", "startup", "mobile"}.
Note that for the synthetic dataset, the formula and configs indexes are determined inside the script.

**3)** Create the results csv files:

```bash generate_results_csv.sh <dataset_name1> <dataset_name2> ...```

where dataset_name1, dataset_name2, ... are the names of the datasets.

********************************************************************************
********************************************************************************

