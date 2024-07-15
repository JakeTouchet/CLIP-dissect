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

# Set up argument parser for command-line options
parser = argparse.ArgumentParser(description='CLIP-Dissect')

# Define command-line arguments
parser.add_argument("--clip_model", type=str, default="ViT-B/16", 
                    choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'],
                    help="Which CLIP-model to use")
parser.add_argument("--target_model1", default="finetune_resnet_gelu", type=str, 
                    help="First model to dissect")
parser.add_argument("--target_model2", type=str, default="gradnorm_resnet_gelu", 
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
parser.add_argument("--vis_results_dir", type=str, default="../vis_results", help="where to save visualization results")
parser.add_argument("--pool_mode", type=str, default="avg", help="Aggregation function for channels, max or avg")
parser.add_argument("--similarity_fn", type=str, default="soft_wpmi", 
                    choices=["soft_wpmi", "wpmi", "rank_reorder", "cos_similarity", "cos_similarity_cubed"])

def process_model(args, target_model, similarities, target_feats, words, pil_data, neurons_to_check):
    """
    Process a single model, creating visualizations for specified neurons.
    
    Args:
        args: Command-line arguments
        target_model: Name of the model being processed
        similarities: Similarity scores between neurons and concepts
        target_feats: Features extracted from the target model
        words: List of concepts
        pil_data: Image data
        neurons_to_check: List of neuron indices to visualize
    
    Returns:
        fig: Matplotlib figure containing the visualizations
        top_concepts: Dictionary mapping neuron indices to their top concept
    """
    # Get top 5 activating images for each neuron
    top_vals, top_ids = torch.topk(target_feats.float(), k=5, dim=0)
    
    # Create a figure with subplots for each neuron
    fig = plt.figure(figsize=[20, len(neurons_to_check)*3])  # Increased figure size
    subfigs = fig.subfigures(nrows=len(neurons_to_check), ncols=1)
    
    top_concepts = {}
    
    for j, orig_id in enumerate(neurons_to_check):
        # Get top 5 similar concepts for the neuron
        vals, ids = torch.topk(similarities[orig_id], k=5, largest=True)
        subfig = subfigs[j]
        
        # Store the top concept for this neuron
        top_concepts[orig_id] = words[int(ids[0])]
        
        # Set the title for each neuron's subplot
        subfig.suptitle(f"{target_model} - Neuron {int(orig_id)}:\nCLIP-Dissect: {top_concepts[orig_id]}", 
                        fontsize=12, y=1.05)
        
        # Create subplots for the top 5 activating images
        axs = subfig.subplots(nrows=1, ncols=5)
        for i, (top_id, val) in enumerate(zip(top_ids[:, orig_id], top_vals[:, orig_id])):
            im, label = pil_data[top_id]
            im = im.resize([375,375])
            axs[i].imshow(im)
            axs[i].set_title(f"Value: {val:.4f}", fontsize=10)
            axs[i].axis('off')
    
    plt.tight_layout()  # Adjust the layout to prevent overlap
    return fig, top_concepts

def compare_models(args, model1_concepts, model2_concepts, model1_feats, model2_feats, words, pil_data):
    """
    Create a figure comparing the two models, showing changed descriptions and top activating images.
    
    Args:
        args: Command-line arguments
        model1_concepts: Dictionary of top concepts for model1
        model2_concepts: Dictionary of top concepts for model2
        model1_feats: Features extracted from model1
        model2_feats: Features extracted from model2
        words: List of concepts
        pil_data: Image data
    
    Returns:
        fig: Matplotlib figure containing the comparison
    """
    changed_neurons = [n for n in model1_concepts if model1_concepts[n] != model2_concepts[n]]
    num_changed = len(changed_neurons)
    
    # Select up to 7 neurons with changed descriptions
    neurons_to_show = changed_neurons[:7]
    
    fig = plt.figure(figsize=[20, 5 + len(neurons_to_show)*6])
    
    # Add text about number of changed descriptions
    fig.text(0.5, 0.98, f"Number of neurons with changed descriptions: {num_changed}", 
             ha='center', va='top', fontsize=16)
    
    for i, neuron in enumerate(neurons_to_show):
        # Get top 5 activating images for this neuron in both models
        top_vals1, top_ids1 = torch.topk(model1_feats[:, neuron].float(), k=5)
        top_vals2, top_ids2 = torch.topk(model2_feats[:, neuron].float(), k=5)
        
        # Create two rows of subplots for this neuron
        axs1 = fig.add_subplot(len(neurons_to_show), 2, 2*i+1)
        axs2 = fig.add_subplot(len(neurons_to_show), 2, 2*i+2)
        
        axs1.text(0, 1.1, f"{args.target_model1} - Neuron {neuron}: {model1_concepts[neuron]}", 
                  transform=axs1.transAxes, fontsize=12)
        axs2.text(0, 1.1, f"{args.target_model2} - Neuron {neuron}: {model2_concepts[neuron]}", 
                  transform=axs2.transAxes, fontsize=12)
        
        # Display top activating images for each model
        for j, (id1, id2) in enumerate(zip(top_ids1, top_ids2)):
            im1, _ = pil_data[id1]
            im2, _ = pil_data[id2]
            
            ax1 = fig.add_subplot(len(neurons_to_show), 10, 10*i + j + 1)
            ax2 = fig.add_subplot(len(neurons_to_show), 10, 10*i + j + 6)
            
            ax1.imshow(im1.resize([100,100]))
            ax2.imshow(im2.resize([100,100]))
            
            ax1.axis('off')
            ax2.axis('off')
    
    plt.tight_layout()
    return fig

def main(args):
    """
    Main function to process both target models and generate visualizations.
    """
    target_layer = args.target_layer
    similarity_fn = eval(f"similarity.{args.similarity_fn}")

    # Load concept words from file
    with open(args.concept_set, 'r') as f: 
        words = f.read().split('\n')

    # Load image data
    pil_data = data_utils.get_data(args.d_probe)

    # Create vis_results directory if it doesn't exist
    os.makedirs(args.vis_results_dir, exist_ok=True)

    model_results = {}

    # Process each target model
    for target_model in [args.target_model1, args.target_model2]:
        # Generate save names for activations
        print(f"Processing {target_model}...")
        save_names = utils.get_save_names(clip_name=args.clip_model, target_name=target_model,
                                    target_layer=target_layer, d_probe=args.d_probe,
                                    concept_set=args.concept_set, pool_mode=args.pool_mode,
                                    save_dir=args.activation_dir)
        target_save_name, clip_save_name, text_save_name = save_names
        
        # Compute similarities between neurons and concepts
        print("Computing similarities...")
        similarities, target_feats = utils.get_similarity_from_activations(target_save_name, clip_save_name, 
                                                                    text_save_name, similarity_fn)

        # Process the model and generate visualizations
        print("Processing model...")
        fig, top_concepts = process_model(args, target_model, similarities, target_feats, words, pil_data, 
                                          list(range(similarities.shape[0][:10])))
        
        # Save the figure
        plt.savefig(os.path.join(args.vis_results_dir, f'{target_model}_neuron_activations.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free up memory

        model_results[target_model] = {'top_concepts': top_concepts, 'target_feats': target_feats}

    # Create and save the comparison figure
    print("Creating comparison figure...")
    comparison_fig = compare_models(args, model_results[args.target_model1]['top_concepts'], 
                                    model_results[args.target_model2]['top_concepts'],
                                    model_results[args.target_model1]['target_feats'],
                                    model_results[args.target_model2]['target_feats'],
                                    words, pil_data)
    
    plt.savefig(os.path.join(args.vis_results_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(comparison_fig)

    print(f"Plots saved in {args.vis_results_dir}")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)