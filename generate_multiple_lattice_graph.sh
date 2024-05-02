#!/bin/zsh

formulas=(1 2 3 4 5);
configs=(1 2 3 4);
edge_sampling_ratio_list=(0.5)

# Check if the dataset names are provided as command-line arguments
if [ $# -eq 0 ]; then
    echo "Please Insert Dataset Names as Command Line Arguments"
    exit 1
fi
dataset_name_list=("$@")
# dataset_name_list=("synthetic" "loan" "startup" "mobile")

for dataset_name in "${dataset_name_list[@]}"; do
  if [ $dataset_name != "synthetic" ]; then
    for edge_sampling_ratio in "${edge_sampling_ratio_list[@]}"; do
      echo "Generating lattice for dataset $dataset_name, edge_sampling_ratio $edge_sampling_ratio"
      python lattice_graph_generator_multiprocessing.py --data_name $dataset_name \
            --edge_sampling_ratio $edge_sampling_ratio
      echo "***************************************************************************************"
    done
  else
      for j in "${configs[@]}"; do
        for i in "${formulas[@]}"; do
          echo "Generating data for formula $i and config $j"
          python logical_synthetic_data_generator.py --formula $i --config $j
          for edge_sampling_ratio in "${edge_sampling_ratio_list[@]}"; do
            echo "Generating lattice for formula $i, config $j, edge_sampling_ratio $edge_sampling_ratio"
            python lattice_graph_generator_multiprocessing.py --formula $i --config $j \
                  --edge_sampling_ratio $edge_sampling_ratio --data_name $dataset_name
            echo "***************************************************************************************"
          done
        done
      done
  fi
done
