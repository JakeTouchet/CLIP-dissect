import os
import argparse
import datetime
import json
import pandas as pd
import torch
import utils
import similarity
import matplotlib
from matplotlib import pyplot as plt
import data_utils

parser = argparse.ArgumentParser(description='CLIP-Dissect')

parser.add_argument("--clip_model", type=str, default="ViT-B/16", 
                    choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'],
                    help="Which CLIP-model to use")
parser.add_argument("--target_model1", type=str, required=True, 
                    help="First model to dissect")
parser.add_argument("--target_model2", type=str, required=True, 
                    help="Second model to dissect")
parser.add_argument("--target_layer", type=str, default="layer4",
                    help="Which layer neurons to describe")
parser.add_argument("--d_probe", type=str, default="broden", 
                    choices=["imagenet_broden", "cifar100_val", "imagenet_val", "broden", "imagenet_broden"])
parser.add_argument("--concept_set", type=str, default="data/20k.txt", help="Path to txt file containing concept set")
parser.add_argument("--batch_size", type=int, default=200, help="Batch size when running CLIP/target model")
parser.add_argument("--device", type=str, default="cuda", help="whether to use GPU/which gpu")
parser.add_argument("--activation_dir", type=str, default="saved_activations", help="where to save activations")
parser.add_argument("--result_dir", type=str, default="results", help="where to save results")
parser.add_argument("--pool_mode", type=str, default="avg", help="Aggregation function for channels, max or avg")
parser.add_argument("--similarity_fn", type=str, default="soft_wpmi", 
                    choices=["soft_wpmi", "wpmi", "rank_reorder", "cos_similarity", "cos_similarity_cubed"])

def process_model(args, target_model, similarities, target_feats, words, pil_data, neurons_to_check):
    top_vals, top_ids = torch.topk(target_feats.float(), k=5, dim=0)
    
    fig = plt.figure(figsize=[20, len(neurons_to_check)*3])  # Increased figure size
    subfigs = fig.subfigures(nrows=len(neurons_to_check), ncols=1)
    
    for j, orig_id in enumerate(neurons_to_check):
        vals, ids = torch.topk(similarities[orig_id], k=5, largest=True)
        subfig = subfigs[j]
        
        # Use two lines for the title to prevent overlap
        subfig.suptitle(f"{target_model} - Neuron {int(orig_id)}:\nCLIP-Dissect: {words[int(ids[0])]}", 
                        fontsize=12, y=1.05)
        
        axs = subfig.subplots(nrows=1, ncols=5)
        for i, (top_id, val) in enumerate(zip(top_ids[:, orig_id], top_vals[:, orig_id])):
            im, label = pil_data[top_id]
            im = im.resize([375,375])
            axs[i].imshow(im)
            axs[i].set_title(f"Value: {val:.4f}", fontsize=10)
            axs[i].axis('off')
    
    plt.tight_layout()  # Adjust the layout to prevent overlap
    return fig

def main(args):
    target_layer = args.target_layer
    similarity_fn = eval(f"similarity.{args.similarity_fn}")

    with open(args.concept_set, 'r') as f: 
        words = f.read().split('\n')

    pil_data = data_utils.get_data(args.d_probe)

    for target_model in [args.target_model1, args.target_model2]:
        save_names = utils.get_save_names(clip_name=args.clip_model, target_name=target_model,
                                    target_layer=target_layer, d_probe=args.d_probe,
                                    concept_set=args.concept_set, pool_mode=args.pool_mode,
                                    save_dir=args.activation_dir)
        target_save_name, clip_save_name, text_save_name = save_names
        similarities, target_feats = utils.get_similarity_from_activations(target_save_name, clip_save_name, 
                                                                    text_save_name, similarity_fn)

        fig = process_model(args, target_model, similarities, target_feats, words, pil_data)
        
        # Save the figure
        plt.savefig(f'{target_model}_neuron_activations.png', dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free up memory

    print(f"Plots saved for {args.target_model1} and {args.target_model2}")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)