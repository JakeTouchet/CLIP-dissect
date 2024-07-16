#!/bin/bash

# This script runs describe_neurons.py on specified models and layer,
target_models="0005,0010,0020,model_best"
layer="layer4"

for model in $(echo $target_models | sed "s/,/ /g")
do
    python describe_neurons.py --target_model $model --target_layers $layer
done
