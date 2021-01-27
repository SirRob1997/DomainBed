import torch
import torch.nn.functional as F
import os
import json
import seaborn as sns; sns.set()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
from matplotlib.text import Text
from matplotlib.patches import Rectangle



PLOT_PATH = "plots/heatmaps"


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def generate_plot(l2_distances, cosine_distances, trial_index, num_prototypes_per_class, num_classes):

        MAPPING_DICT = {"P": 0, "A": 1, "C": 2, "S": 3}
        ROWS = ['L2 Distance', 'Cosine Distance']

        fig, axes = plt.subplots(2,5, sharex=True, sharey=True)
        for env_name in l2_distances:
            l2_dist_matrix = l2_distances[env_name]
            cbar_flag = False
            mask = torch.tril(l2_dist_matrix)
            sns.heatmap(l2_dist_matrix, ax=axes[0, MAPPING_DICT[env_name]], mask=mask.numpy(), linewidths=0.2, square=True, cbar=cbar_flag, cmap="Blues", xticklabels=False, yticklabels=False)

        for env_name in cosine_distances:
            cosine_dist_matrix = cosine_distances[env_name]
            cbar_flag = False
            mask = torch.tril(cosine_dist_matrix)
            sns.heatmap(cosine_dist_matrix, ax=axes[1, MAPPING_DICT[env_name]], mask=mask.numpy(), linewidths=0.2, square=True, cbar=cbar_flag, cmap="Blues", xticklabels=False, yticklabels=False)

        for height_i in range(2):
            for width_i in range(4):
                for class_index in range(num_classes):
                    x = class_index * num_prototypes_per_class
                    y = class_index * num_prototypes_per_class
                    axes[height_i, width_i].add_patch(Rectangle((x, y), num_prototypes_per_class, num_prototypes_per_class, fill=False, edgecolor='red', lw=0.2))

        for ax, col in zip(axes[0], MAPPING_DICT.keys()):
            ax.set_title(col, size='small')

        for ax, row in zip(axes[:,0], ROWS):
            ax.set_ylabel(row, size='small')

        file_name = f'ProDropIncorrectWeight-1.0WithSCdrop_f0.5_trial{trial_index}.pdf'
        print("Saving Figure", file_name)
        fig.savefig(os.path.join(PLOT_PATH, file_name))
    
if __name__ == "__main__":

    with open('analysis/ProDropIncorrectWeight-1.0WithSCdrop_f0.5.json') as json_file:
        trial_seeds = json.load(json_file) 

    for trial_index in trial_seeds:
        l2_distances = {}
        cosine_distances = {}
        for env_ind, environment_path in trial_seeds[trial_index].items():
            parameters = torch.load(environment_path)
            hyperparams = parameters["model_hparams"]
            num_prototypes = hyperparams["num_prototypes_per_class"] * parameters["model_num_classes"] 
            prototype_vectors = parameters["model_dict"]["pplayer.prototype_vectors"].view(num_prototypes, -1)
            pairwise_distance_cosine = cosine_distance_torch(prototype_vectors)
            pairwise_distance_l2 = torch.norm(prototype_vectors[:, None] - prototype_vectors, dim=2, p=2)
            l2_distances[env_ind] = pairwise_distance_l2
            cosine_distances[env_ind] = pairwise_distance_cosine
        generate_plot(l2_distances, cosine_distances,trial_index, hyperparams["num_prototypes_per_class"], parameters["model_num_classes"])
