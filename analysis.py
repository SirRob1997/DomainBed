
import argparse
import functools
import glob
import pickle
import itertools
import json
import os
import random
import sys

import torch
import torch.nn.functional as F
import os
import json
import seaborn as sns; sns.set()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import QuadMesh
from matplotlib.text import Text
from matplotlib.patches import Rectangle


from domainbed import datasets
from domainbed import algorithms
from domainbed.lib import misc, reporting
from domainbed import model_selection
from domainbed.lib.query import Q



SELECTION_METHODS = {
        #model_selection.IIDAccuracySelectionMethod: "train",
        #model_selection.LeaveOneOutSelectionMethod: "leave_out",
        model_selection.OracleSelectionMethod: "oracle"
    }



PLOT_PATH = "plots/heatmaps"


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def generate_indiv_plot(distances, cosine, trial_index, num_prototypes_per_class, num_classes, path):

        MAPPING_DICT = {"P": 0, "A": 1, "C": 2, "S": 3}
        if cosine:
            vmin = 0
            vmax = 2
        else:
            vmin = 2
            vmax = 15

        # PLOTS
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        cbar_ax = fig.add_axes([.86, 0.1, .03, 0.8])
        for env_name in distances:
            dist_matrix = distances[env_name]
            cbar_flag = True if MAPPING_DICT[env_name]==0 else False
            mask = torch.tril(dist_matrix)
            sns.heatmap(dist_matrix, ax=axes[MAPPING_DICT[env_name]//2, MAPPING_DICT[env_name]%2], mask=mask.numpy(), linewidths=0.2, square=True, cbar=cbar_flag, cbar_ax=cbar_ax if cbar_flag else None, cmap="Blues", xticklabels=False, yticklabels=False, vmin=vmin, vmax=vmax)
        
        for height_i in range(2):
            for width_i in range(2):
                for class_index in range(num_classes):
                    x = class_index * num_prototypes_per_class
                    y = class_index * num_prototypes_per_class
                    axes[height_i, width_i].add_patch(Rectangle((x, y), num_prototypes_per_class, num_prototypes_per_class, fill=False, edgecolor='red', lw=0.2))

        for ax, col in zip(axes.flat, MAPPING_DICT.keys()):
            ax.set_title(col, size='small')

        splitted = path.split('/')
        run_name = splitted[-3] + splitted[-2] + (splitted[-1].split('.')[-2])
        if cosine:
            fig.text(0.04, 0.5, 'Cosine Distance', va='center', rotation='vertical')
            file_name = run_name + f'_trial{trial_index}_cosine.pdf'
        else:
            fig.text(0.04, 0.5, 'L2 Distance', va='center', rotation='vertical')
            file_name = run_name + f'_trial{trial_index}_l2.pdf'
        fig.tight_layout(rect=[0, 0, .9, 1])
        print("Saving Figure", file_name, "at", PLOT_PATH)
        fig.savefig(os.path.join(PLOT_PATH, file_name))


def generate_joint_plot(l2_distances, cosine_distances, trial_index, num_prototypes_per_class, num_classes, path):
    #Options
    params = {'text.usetex' : True,
          'font.size' : 11,
          'font.family' : 'lmodern',
          'axes.labelsize': 11,
          'legend.fontsize': 11,
          'text.latex.preamble': r"\usepackage{lmodern}" 
          }
    plt.rcParams.update(params) 

    MAPPING_DICT = {"P": 0, "A": 1, "C": 2, "S": 3}
    ROWS = ['L2 Distance', 'Cosine Distance']
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=False, figsize=(16, 8),
                             gridspec_kw={'width_ratios': [10, 10, 10, 10, 1]})
    shax = axes[0, 0].get_shared_x_axes()
    shay = axes[0, 0].get_shared_y_axes()
    for ax in axes[:, :-1].ravel():
        shax.join(axes[0, 0], ax)
        shay.join(axes[0, 0], ax)

    for env_name in l2_distances:
        l2_dist_matrix = l2_distances[env_name]
        mask = torch.tril(l2_dist_matrix)
        cbar_flag = True if MAPPING_DICT[env_name] == 3 else False
        sns.heatmap(l2_dist_matrix, ax=axes[0, MAPPING_DICT[env_name]], mask=mask.numpy(), linewidths=0.2, square=True,
                        cbar=cbar_flag, cbar_ax=axes[0, -1], cmap="Blues", xticklabels=False, yticklabels=False, vmin=0, vmax=14)

    for env_name in cosine_distances:
        cosine_dist_matrix = cosine_distances[env_name]
        mask = torch.tril(cosine_dist_matrix)
        cbar_flag = True if MAPPING_DICT[env_name] == 3 else False
        sns.heatmap(cosine_dist_matrix, mask=mask.numpy(), ax=axes[1, MAPPING_DICT[env_name]], linewidths=0.2, square=True,
                        cbar=cbar_flag, cbar_ax=axes[1, -1], cmap="Blues", xticklabels=False, yticklabels=False, vmin=0, vmax=1)

    for height_i in range(2):
        for width_i in range(4):
            for class_index in range(num_classes):
                x = class_index * num_prototypes_per_class
                y = class_index * num_prototypes_per_class
                axes[height_i, width_i].add_patch(Rectangle((x, y), num_prototypes_per_class, num_prototypes_per_class, fill=False, edgecolor='red', lw=0.2))

    for ax, col in zip(axes[0], MAPPING_DICT.keys()):
        ax.set_title(col, fontweight='bold', fontsize=25)

    for ax, row in zip(axes[:,0], ROWS):
        ax.set_ylabel(row, fontweight='bold', fontsize=25)

    splitted = path.split('/')
    run_name = splitted[-3] + splitted[-2] + (splitted[-1].split('.')[-2])
    file_name = run_name + f'_trial{trial_index}.pdf'
    print("Saving Figure", file_name, "at", PLOT_PATH)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_PATH, file_name), dpi=1000, bbox_inches='tight')

def generate_plots(paths, args):
    for path in paths:
        with open(path) as json_file:
            trial_seeds = json.load(json_file)

        for trial_index in trial_seeds:
            l2_distances = {}
            cosine_distances = {}
            l2_path = os.path.splitext(path)[0] + f"_l2_distances_trial{trial_index}.pt"
            cosine_path = os.path.splitext(path)[0] + f"_cosine_distances_trial{trial_index}.pt"
            if not args.skip_data_load:
                for env_ind, environment_path in trial_seeds[trial_index].items():
                    parameters = torch.load(os.path.join(environment_path, "model.pkl"))
                    hyperparams = parameters["model_hparams"]
                    num_prototypes = hyperparams["num_prototypes_per_class"] * parameters["model_num_classes"] 
                    prototype_vectors = parameters["model_dict"]["pplayer.prototype_vectors"].view(num_prototypes, -1)
                    pairwise_distance_cosine = cosine_distance_torch(prototype_vectors)
                    pairwise_distance_l2 = torch.norm(prototype_vectors[:, None] - prototype_vectors, dim=2, p=2)
                    l2_distances[env_ind] = pairwise_distance_l2
                    cosine_distances[env_ind] = pairwise_distance_cosine
                print("Saved", l2_path)
                print("Saved", cosine_path)
                torch.save(l2_distances, l2_path)
                torch.save(cosine_distances, cosine_path)
            else:
                l2_distances = torch.load(l2_path)
                cosine_distances = torch.load(cosine_path)

            generate_joint_plot(l2_distances, cosine_distances, trial_index, 10, 7, path)
            #generate_indiv_plot(l2_distances, False, trial_index, hyperparams["num_prototypes_per_class"], parameters["model_num_classes"], path)
            #generate_indiv_plot(cosine_distances, True, trial_index, hyperparams["num_prototypes_per_class"], parameters["model_num_classes"], path)


def generate_jsons(records, args):
    TEST_ENV_LOOKUP = {0: 'A', 1: 'C', 2: 'P', 3: 'S'}
    json_paths = []

    for selection_method in SELECTION_METHODS.keys():
        results_dic = {}
        for test_env in range(4):
            tmp_records = records.filter(
            lambda r:
                r['dataset'] == args.dataset and
                r['algorithm'] == args.algorithm and
                r['test_env'] == test_env
             )
            for group in tmp_records:
                best_hparams = selection_method.hparams_accs(group['records'])[0]
                run_acc = best_hparams[0]
                used_parameters = best_hparams[1]
                #if test_env == 1 and group['trial_seed']==2:
                #    print(used_parameters[14])
                #    print("\n")
                #    input()
                #for k, v in sorted(used_parameters[0]['hparams'].items()):
                #    print('\t\t\t{}: {}'.format(k, v))
                output_dirs = used_parameters.select('args.output_dir').unique()
                if group['trial_seed'] in results_dic:
                    results_dic[group['trial_seed']][TEST_ENV_LOOKUP[test_env]] = output_dirs[0]
                else:
                    results_dic[group['trial_seed']] = {TEST_ENV_LOOKUP[test_env]: output_dirs[0]}
        file_path = os.path.join(args.input_dir, SELECTION_METHODS[selection_method] + "_validation.json")
        print("Saving file at", file_path)
        with open(file_path, 'w') as file:
            json.dump(results_dic, file, indent=4)
        json_paths.append(file_path)

    return json_paths

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analsis")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--algorithm', required=True)
    parser.add_argument('--skip_data_load', action='store_true', default=False)

    args = parser.parse_args()

    if args.skip_data_load:
	    paths = [os.path.join(args.input_dir, SELECTION_METHODS[selection_method] + "_validation.json") for selection_method in SELECTION_METHODS.keys()]
    else:
        paths = generate_jsons(records, args)
        records = reporting.load_records(args.input_dir)
        print("Total records:", len(records))
        records = reporting.get_grouped_records(records)
    generate_plots(paths, args)
