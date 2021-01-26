import torch
import torch.nn.functional as F
import os
import json
import seaborn as sns


PLOT_PATH = "plots/heatmaps"


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def generate_plot(l2_distances, cosine_distances, trial_index):
        heatmap_ax = sns.heatmap(l2_distances[0])
        fig = heatmap_ax.get_figure()
        fig.savefig(os.path.join(PLOT_PATH, f'ProDropIncorrectWeight-1.0WithSCdrop_f0.5_trial{trial_index}.pdf'))
    
if __name__ == "__main__":

    with open('analysis/ProDropIncorrectWeight-1.0WithSCdrop_f0.5.json') as json_file:
        trial_seeds = json.load(json_file) 

    for trial_index in trial_seeds:
        l2_distances = []
        cosine_distances = []
        for env_ind, environment_path in trial_seeds[trial_index].items():
            parameters = torch.load(environment_path)
            hyperparams = parameters["model_hparams"]
            num_prototypes = hyperparams["num_prototypes_per_class"] * parameters["model_num_classes"] 
            prototype_vectors = parameters["model_dict"]["pplayer.prototype_vectors"].view(num_prototypes, -1)
            pairwise_distance_l2 = torch.norm(prototype_vectors[:, None] - prototype_vectors, dim=2, p=2)
            pairwise_distance_cosine = cosine_distance_torch(prototype_vectors)
            l2_distances.append(pairwise_distance_l2)
            cosine_distances.append(pairwise_distance_cosine)
        generate_plot(l2_distances, cosine_distances,trial_index)
