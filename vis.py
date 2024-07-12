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
parser.add_argument("--target_model", type=str, default="resnet50", 
                   help=""""Which model to dissect, supported options are pretrained imagenet models from
                        torchvision and resnet18_places""")
parser.add_argument("--target_layer", type=str, default="layer4",
                    help="""Which layer neurons to describe. String list of layer names to describe, separated by comma(no spaces). 
                          Follows the naming scheme of the Pytorch module used""")
parser.add_argument("--d_probe", type=str, default="broden", 
                    choices = ["imagenet_broden", "cifar100_val", "imagenet_val", "broden", "imagenet_broden"])
parser.add_argument("--concept_set", type=str, default="data/20k.txt", help="Path to txt file containing concept set")
parser.add_argument("--batch_size", type=int, default=200, help="Batch size when running CLIP/target model")
parser.add_argument("--device", type=str, default="cuda", help="whether to use GPU/which gpu")
parser.add_argument("--activation_dir", type=str, default="saved_activations", help="where to save activations")
parser.add_argument("--result_dir", type=str, default="results", help="where to save results")
parser.add_argument("--pool_mode", type=str, default="avg", help="Aggregation function for channels, max or avg")
parser.add_argument("--similarity_fn", type=str, default="soft_wpmi", choices=["soft_wpmi", "wpmi", "rank_reorder", 
                                                                               "cos_similarity", "cos_similarity_cubed"])



def main(args):
    target_layer = "layer4"
    similarity_fn = eval("similarity.{}".format(args.similarity_fn))
    save_names = utils.get_save_names(clip_name = args.clip_model, target_name = args.target_model,
                                target_layer = target_layer, d_probe = args.d_probe,
                                concept_set = args.concept_set, pool_mode = args.pool_mode,
                                save_dir = args.activation_dir)
    target_save_name, clip_save_name, text_save_name = save_names
    similarities, target_feats = utils.get_similarity_from_activations(target_save_name, clip_save_name, 
                                                                text_save_name, similarity_fn)

    with open(args.concept_set, 'r') as f: 
        words = (f.read()).split('\n')

    pil_data = data_utils.get_data(args.d_probe)

    neurons_to_check = torch.sort(torch.max(similarities, dim=1)[0], descending=True)[1][0:10]
    top_vals, top_ids = torch.topk(target_feats.float(), k=5, dim=0)
    font_size=14
    font = {'size'   : font_size}

    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=[10, len(neurons_to_check)*2])#constrained_layout=True)
    subfigs = fig.subfigures(nrows=len(neurons_to_check), ncols=1)
    for j, orig_id in enumerate(neurons_to_check):
        vals, ids = torch.topk(similarities[orig_id], k=5, largest=True)
            
        subfig = subfigs[j]
        subfig.text(0.13, 0.96, "Neuron {}:".format(int(orig_id)), size=font_size)
        subfig.text(0.27, 0.96, "CLIP-Dissect:", size=font_size)
        subfig.text(0.4, 0.96, words[int(ids[0])], size=font_size)
        axs = subfig.subplots(nrows=1, ncols=5)
        for i, top_id in enumerate(top_ids[:, orig_id]):
            im, label = pil_data[top_id]
            im = im.resize([375,375])
            axs[i].imshow(im)
            axs[i].axis('off')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)