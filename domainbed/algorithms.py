# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import itertools

import copy
import numpy as np

from domainbed import datasets
from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches, cosine_distance_torch

ALGORITHMS = [
    'ERM',
    'IRM',
    'GroupDRO',
    'Mixup',
    'MLDG',
    'CORAL',
    'MMD',
    'DANN', 
    'CDANN', 
    'MTL', 
    'SagNet',
    'ARM',
    'VREx',
    'RSC',
    'ProDrop',
    'ProDropEnsamble'
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=self.featurizer.n_outputs, out_channels=self.featurizer.n_outputs, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.featurizer.n_outputs, out_channels=self.featurizer.n_outputs, kernel_size=1),
                nn.Sigmoid(),
         )

        # Remove AvgPool, Flatten and Droput for ResNet
        if self.featurizer.__class__.__name__ == "ResNet":
            self.featurizer.network.avgpool = networks.Identity()
            self.featurizer.flattenLayer = networks.Identity()
            self.featurizer.dropout = networks.Identity()

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(self.featurizer, self.add_on_layers, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches):
        self.update_count += 1
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        x = self.add_on_layers(x)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class ProDrop(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ProDrop, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.num_classes = num_classes
        self.num_prototypes_per_class = hparams['num_prototypes_per_class'] 
        self.num_prototypes = self.num_prototypes_per_class  * num_classes
        self.prototype_width = hparams['prototype_width']
        self.prototype_height = hparams['prototype_height']
        self.prototype_shape = (self.num_prototypes, self.featurizer.n_outputs, self.prototype_height, self.prototype_width)
        self.additional_losses = hparams['additional_losses']
        self.self_challenging = hparams['self_challenging']
        self.drop_b = hparams['drop_b']
        self.drop_f = hparams['drop_f']
        self.ce_factor = hparams['ce_factor']
        self.cl_factor = hparams['cl_factor']
        self.sep_factor = hparams['sep_factor']
        #self.l1_factor = hparams['l1_factor']
        #self.cpt_factor = hparams['cpt_factor']
        self.intra_factor = hparams['intra_factor']
        self.end_to_end = hparams['end_to_end']
        self.negative_weight = hparams['negative_weight']

        self.pplayer = networks.PPLayer(self.prototype_shape, num_classes)
        self.classifier = nn.Linear(self.num_prototypes, num_classes, bias=False)
        self._initialize_weights()

        # Remove AvgPool, Flatten and Droput for ResNet
        if self.featurizer.__class__.__name__ == "ResNet":
            self.featurizer.network.avgpool = networks.Identity()
            self.featurizer.flattenLayer = networks.Identity()
            self.featurizer.dropout = networks.Identity()

        self.network = nn.Sequential(self.featurizer, self.pplayer, self.classifier)

        if self.hparams['freeze_classifier']: 
            self.freeze_parameters(self.classifier)
            self.frozen_classifier = True

        if self.end_to_end:
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])
        else:
            self.register_buffer('update_count', torch.tensor([0]))
            self.warmup_steps = self.hparams['warmup_steps']
            self.optimize_classifier = self.hparams['optimize_classifier']
            self.cooldown_steps = self.hparams['cooldown_steps']
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])
            self.pplayer_optimizer = torch.optim.Adam(
                self.pplayer.parameters(),
                lr=self.hparams["pp_lr"],
                weight_decay=self.hparams['pp_weight_decay'])
            self.classifier_optimizer = torch.optim.Adam(
                self.classifier.parameters(),
                lr=self.hparams["cl_lr"])

    def freeze_parameters(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        positive_one_weights_locations = torch.t(self.pplayer.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.classifier.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in self.pplayer.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=self.negative_weight)
 
    def calculate_intra_loss_l2(self, prototypes):
        loss = 0
        for class_i in range(self.num_classes):
            curr_prototypes = prototypes[self.pplayer.prototype_class_identity[:, class_i].bool(), :]
            distances = F.pdist(prototypes, p=2)
            loss += torch.mean(distances)
        return loss / self.num_classes


    def calculate_intra_loss_cosine(self, prototypes):
        loss = 0
        for class_i in range(self.num_classes):
            curr_prototypes = prototypes[self.pplayer.prototype_class_identity[:, class_i].bool(), :]
            distances = cosine_distance_torch(prototypes)
            loss += torch.mean(distances)
        return loss / self.num_classes


    def update(self, minibatches):
        self.update_count += 1
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_o = F.one_hot(all_y, self.num_classes)
        features = self.featurizer(all_x)
        prot_activations = self.pplayer(features)
        outputs = self.classifier(prot_activations)
        if self.self_challenging:
            mask_p = torch.t(self.pplayer.prototype_class_identity[:, all_y]).cuda().bool() # prototypes which correspond to the class
            reduced_activations = prot_activations * mask_p
            quantile_f = torch.quantile(reduced_activations, 1 - (self.drop_f * (self.num_prototypes_per_class / self.num_prototypes)), dim=1, keepdim=True)
            mask_f = reduced_activations.lt(quantile_f) # 0 for prototype activations to apply masking, highest values in prot_activations, i.e. the highest similarity prototypes
            all_s = F.softmax(outputs, dim=1)
            before_vector = (all_s * all_o).sum(1)
            before_vector = torch.where(before_vector > 0, before_vector, torch.zeros(before_vector.shape).cuda())
            quantile_b = torch.quantile(before_vector, 1 - self.drop_b)
            mask_b = before_vector.lt(quantile_b).view(-1,1).repeat(1, prot_activations.shape[1]) # 0 for samples to apply masking, highest values in before_vector i.e. highest confidence on correct class 
            mask = torch.logical_or(mask_f, mask_b).float()
            muted_outputs = self.classifier(prot_activations * mask)
            #print("Masked out values for this batch:", (mask.shape[1]) - torch.count_nonzero(mask, dim=1))
            ce_loss = F.cross_entropy(muted_outputs, all_y)
        else:
            ce_loss = F.cross_entropy(outputs, all_y)

        # Decision on whether we want to add other losses to the CE loss
        if self.additional_losses:
            max_dist = (self.prototype_shape[1] * self.prototype_shape[2] * self.prototype_shape[3])

            # calculate cluster cost
            prototypes_of_correct_class = torch.t(self.pplayer.prototype_class_identity[:, all_y]).cuda() # [N, num_prototypes]
            inverted_distances, _ = torch.max((max_dist - self.pplayer.min_distances) * prototypes_of_correct_class, dim=1) # [N]
            cluster_loss = torch.mean(max_dist - inverted_distances)

            # calculate separation cost
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            inverted_distances_to_nontarget_prototypes, _ = torch.max((max_dist - self.pplayer.min_distances) * prototypes_of_wrong_class, dim=1)
            separation_loss = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

            # calculate the intra class prototype distance
            reshaped_prototypes = self.pplayer.prototype_vectors.view(self.num_prototypes, -1)
            intra_loss = self.calculate_intra_loss_l2(reshaped_prototypes) + self.calculate_intra_loss_cosine(reshaped_prototypes)

            # get the current cpt loss
            #cpt_loss = self.pplayer.cpt_loss

            # L1 mask
            #l1_mask = 1 - torch.t(self.pplayer.prototype_class_identity).cuda()
            #l1 = (self.classifier.weight * l1_mask).norm(p=1)

            # Overall loss
            loss = self.ce_factor * ce_loss + self.cl_factor * cluster_loss + self.sep_factor * separation_loss + self.intra_factor * intra_loss 
        else:
            loss = ce_loss

        # Decision on whether to train end-to-end or with warmup steps
        if self.end_to_end:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            if self.update_count.item() <= self.warmup_steps:
                self.pplayer_optimizer.zero_grad()
                loss.backward()
                self.pplayer_optimizer.step()
            elif (self.update_count.item() >= 5001 - self.cooldown_steps) and self.optimize_classifier:
                if self.frozen_classifier:
                    self.unfreeze_parameters(self.classifier)
                    self.frozen_classifier = False
                    self.freeze_parameters(self.featurizer)
                    self.freeze_parameters(self.pplayer)
                self.classifier_optimizer.zero_grad()
                loss.backward()
                self.classifier_optimizer.step()
            else:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return {'loss': loss.item(), 'w_intra_loss': intra_loss.item() * self.intra_factor, 'w_ce_loss': self.ce_factor * ce_loss.item(), "w_cl_loss": self.cl_factor * cluster_loss.item(), "w_sep_loss": self.sep_factor * separation_loss.item()}

    def predict(self, x):
        return self.network(x)



class ProDropEnsamble(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ProDropEnsamble, self).__init__(input_shape, num_classes, num_domains, hparams)


        self.num_classes = num_classes
        self.num_domains = num_domains
        self.num_prototypes_per_class = hparams['num_prototypes_per_class']
        self.num_prototypes = self.num_prototypes_per_class * num_classes
        self.prototype_width = hparams['prototype_width']
        self.prototype_height = hparams['prototype_height']
        self.prototype_shape = (
        self.num_prototypes, self.featurizer.n_outputs, self.prototype_height, self.prototype_width)
        self.additional_losses = hparams['additional_losses']
        self.ce_factor = hparams['ce_factor']
        self.cl_factor = hparams['cl_factor']
        self.sep_factor = hparams['sep_factor']
        self.l1_factor = hparams['l1_factor']
        self.cpt_factor = hparams['cpt_factor']
        self.intra_factor = hparams['intra_factor']
        self.end_to_end = hparams['end_to_end']

        # Remove AvgPool, Flatten and Droput for ResNet
        if self.featurizer.__class__.__name__ == "ResNet":
            self.featurizer.network.avgpool = networks.Identity()
            self.featurizer.flattenLayer = networks.Identity()
            self.featurizer.dropout = networks.Identity()

        self.pplayers = [networks.PPLayer(self.prototype_shape, num_classes).cuda() for _ in range(num_domains)]
        self.classifiers = [nn.Linear(self.num_prototypes, num_classes, bias=False).cuda() for _ in range(num_domains)]
        self.aggregation_layer = nn.Linear(num_domains * num_classes, num_classes, bias=False).cuda()

        self._initialize_ensamble_weights()

        if self.hparams['freeze_classifier']:
            for classifier in self.classifiers: 
                self.freeze_parameters(classifier)
                self.frozen_classifier = True

        pplayers_params = list()
        for pplayer in self.pplayers:
            pplayers_params += list(pplayer.parameters()) 

        classifiers_params = list()
        for classifier in self.classifiers:
            classifiers_params += list(classifier.parameters())

        if self.end_to_end:
            self.optimizer = torch.optim.Adam(
                (list(self.featurizer.parameters()) +
                 pplayers_params +
                 classifiers_params) ,
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])
        else:
            self.register_buffer('update_count', torch.tensor([0]))
            self.warmup_steps = self.hparams['warmup_steps']
            self.optimize_classifier = self.hparams['optimize_classifier']
            self.cooldown_steps = self.hparams['cooldown_steps']
            self.optimizer = torch.optim.Adam(
                (list(self.featurizer.parameters()) +
                 pplayers_params +
                classifiers_params),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])
            self.pplayer_optimizer = torch.optim.Adam(
                pplayers_params,
                lr=self.hparams["pp_lr"],
                weight_decay=self.hparams['pp_weight_decay'])
            self.classifier_optimizer = torch.optim.Adam(
                classifiers_params,
                lr=self.hparams["cl_lr"])

    def _initialize_ensamble_weights(self):
        for domain_layer, classifier in zip(self.pplayers, self.classifiers):
            for m in domain_layer.add_on_layers.modules():
                if isinstance(m, nn.Conv2d):
                    # every init technique has an underscore _ in the name
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            self.set_ensemble_classifier_incorrect_connection(domain_layer, classifier, incorrect_strength=-0.5)

    def set_ensemble_classifier_incorrect_connection(self, domain_layer, classifier, incorrect_strength):
        positive_one_weights_locations = torch.t(domain_layer.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        classifier.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def freeze_parameters(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def set_aggregation_weights(self, correct_strength = 1, incorrect_strength = 0, domain=None):
        if domain is not None:
            b_non_selected = torch.zeros(self.num_classes, self.num_classes).repeat(1, domain)
            a_non_selected = torch.zeros(self.num_classes, self.num_classes).repeat(1, self.num_domains - domain - 1)
            weights = torch.eye(self.num_classes)
            weights = torch.cat((b_non_selected, weights, a_non_selected), 1)
        else:
            weights = torch.eye(self.num_classes).repeat(1,self.num_domains)

        if correct_strength != 1:
            weights[weights==1] = correct_strength
        if incorrect_strength != 0:
            weights[weights==0] = incorrect_strength
        self.aggregation_layer.weight.data.copy_(weights)

    def update(self, minibatches):
        features = [self.featurizer(xi) for xi, _ in minibatches]
        targets = [yi for _, yi in minibatches]
        prot_activations = [domain_pplayer(features[domain]) for domain, domain_pplayer in enumerate(self.pplayers)]
        domain_outputs = [classifier(prot_activations[domain]) for domain, classifier in enumerate(self.classifiers)]

        ce_loss = 0
        cluster_loss = 0 
        separation_loss = 0
        for domain, domain_output in enumerate(domain_outputs):
            self.set_aggregation_weights(correct_strength=1, domain=domain)

            # Fill the domain_output vectors up to size num_classes * num_domains by setting all other domain predictions to 0
            b_domain_class = torch.zeros(domain_output.shape[0], self.num_classes * domain).cuda() 
            a_domain_class = torch.zeros(domain_output.shape[0], self.num_classes * (self.num_domains - domain - 1)).cuda() 
            domain_output = torch.cat((b_domain_class, domain_output, a_domain_class), 1)
            output = self.aggregation_layer(domain_output)
            ce_loss += F.cross_entropy(output, targets[domain])

            # Decision on whether we want to add other losses to the CE loss
            if self.additional_losses:
                max_dist = (self.prototype_shape[1] * self.prototype_shape[2] * self.prototype_shape[3])
            
                # calculate cluster cost
                prototypes_of_correct_class = torch.t(self.pplayers[domain].prototype_class_identity[:, targets[domain]]).cuda() 
                inverted_distances, _ = torch.max((max_dist - self.pplayers[domain].min_distances) * prototypes_of_correct_class, dim=1)
                cluster_loss += torch.mean(max_dist - inverted_distances)

                # calculate separation cost
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = torch.max((max_dist - self.pplayers[domain].min_distances) * prototypes_of_wrong_class, dim=1)
                separation_loss += torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)


        if self.additional_losses:
            loss = self.ce_factor * ce_loss + self.cl_factor * cluster_loss + self.sep_factor * separation_loss
        else:
            loss = ce_loss

        # Decision on whether to train end-to-end or with warmup steps
        if self.end_to_end:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            self.update_count += 1
            if self.update_count.item() <= self.warmup_steps:
                self.pplayer_optimizer.zero_grad()
                loss.backward()
                self.pplayer_optimizer.step()
            elif (self.update_count.item() >= 5001 - self.cooldown_steps) and self.optimize_classifier:
                if self.frozen_classifier:
                    self.unfreeze_parameters(self.classifier)
                    self.frozen_classifier = False
                    self.freeze_parameters(self.featurizer)
                    self.freeze_parameters(self.pplayer)
                self.classifier_optimizer.zero_grad()
                loss.backward()
                self.classifier_optimizer.step()
            else:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return {'loss': loss.item()}

    def predict(self, x):
        self.set_aggregation_weights(correct_strength=1/self.num_domains)
        features = self.featurizer(x)
        prot_activations = [domain_pplayer(features) for domain_pplayer in self.pplayers]
        domain_outputs = [classifier(prot_activations[domain]) for domain, classifier in enumerate(self.classifiers)]
        domain_outputs = torch.cat(domain_outputs, 1)
        return self.aggregation_layer(domain_outputs)
        



class ARM(ERM):
    """ Adaptive Risk Minimization (ARM) """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        original_input_shape = input_shape
        input_shape = (1 + original_input_shape[0],) + original_input_shape[1:]
        super(ARM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.context_net = networks.ContextNet(original_input_shape)
        self.support_size = hparams['batch_size']

    def predict(self, x):
        batch_size, c, h, w = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        context = self.context_net(x)
        context = context.reshape((meta_batch_size, support_size, 1, h, w))
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x = torch.cat([x, context], dim=1)
        return self.network(x)


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, conditional, class_balance):

        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.register_buffer('update_count', torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.discriminator = networks.MLP(self.featurizer.n_outputs,
            num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes,
            self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = torch.optim.Adam(
            (list(self.discriminator.parameters()) + 
                list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            betas=(self.hparams['beta1'], 0.9))

        self.gen_opt = torch.optim.Adam(
            (list(self.featurizer.parameters()) + 
                list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams['weight_decay_g'],
            betas=(self.hparams['beta1'], 0.9))

    def update(self, minibatches):
        self.update_count += 1
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((x.shape[0], ), i, dtype=torch.int64, device='cuda')
            for i, (x, y) in enumerate(minibatches)
        ])

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(disc_softmax[:, disc_labels].sum(),
            [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams['grad_penalty'] * grad_penalty

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        if (self.update_count.item() % (1+d_steps_per_g) < d_steps_per_g):

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'disc_loss': disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = (classifier_loss +
                        (self.hparams['lambda'] * -disc_loss))
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {'gen_loss': gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))

class DANN(AbstractDANN):
    """Unconditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DANN, self).__init__(input_shape, num_classes, num_domains,
            hparams, conditional=False, class_balance=False)


class CDANN(AbstractDANN):
    """Conditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CDANN, self).__init__(input_shape, num_classes, num_domains,
            hparams, conditional=True, class_balance=True)


class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches):
        penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                          >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        nll = 0.
        penalty = 0.

        all_x = torch.cat([x for x,y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams['irm_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
            'penalty': penalty.item()}

    
class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(VREx, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches):
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0
        
        nll = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams['vrex_penalty_anneal_iters']:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}

    
class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Mixup, self).__init__(input_shape, num_classes, num_domains,
                                    hparams)

    def update(self, minibatches):
        objective = 0

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"],
                                 self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item()}


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains,
                                        hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, minibatches):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q) / len(minibatches)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf

    TODO: update() has at least one bug, possibly more. Disabling this whole
    algorithm until it gets figured out.
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)

    def update(self, minibatches):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)

            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)

            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)

        That is, when calling .step(), we want grads to be Gi + beta * Gj

        For computational efficiency, we do not compute second derivatives.
        """
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            # fine tune clone-network on task "i"
            inner_net = copy.deepcopy(self.network)

            inner_opt = torch.optim.Adam(
                inner_net.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )

            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(),
                                    inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()

            # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(),
                allow_unused=True)

            # `objective` is populated for reporting purposes
            objective += (self.hparams['mldg_beta'] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(
                        self.hparams['mldg_beta'] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)

        self.optimizer.step()

        return {'loss': objective}

    # This commented "update" method back-propagates through the gradients of
    # the inner update, as suggested in the original MAML paper.  However, this
    # is twice as expensive as the uncommented "update" method, which does not
    # compute second-order derivatives, implementing the First-Order MAML
    # method (FOMAML) described in the original MAML paper.

    # def update(self, minibatches):
    #     objective = 0
    #     beta = self.hparams["beta"]
    #     inner_iterations = self.hparams["inner_iterations"]

    #     self.optimizer.zero_grad()

    #     with higher.innerloop_ctx(self.network, self.optimizer,
    #         copy_initial_weights=False) as (inner_network, inner_optimizer):

    #         for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
    #             for inner_iteration in range(inner_iterations):
    #                 li = F.cross_entropy(inner_network(xi), yi)
    #                 inner_optimizer.step(li)
    #
    #             objective += F.cross_entropy(self.network(xi), yi)
    #             objective += beta * F.cross_entropy(inner_network(xj), yj)

    #         objective /= len(minibatches)
    #         objective.backward()
    #
    #     self.optimizer.step()
    #
    #     return objective


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        if gaussian: 
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)
    
    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches):
        objective = 0
        penalty = 0 
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma']*penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(input_shape, num_classes,
                                          num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference 
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes,
                                         num_domains, hparams, gaussian=False)


class MTL(Algorithm):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning
    (https://arxiv.org/abs/1711.07910)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MTL, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs * 2, num_classes)
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) +\
            list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        self.register_buffer('embeddings',
                             torch.zeros(num_domains,
                                         self.featurizer.n_outputs))

        self.ema = self.hparams['mtl_ema']

    def update(self, minibatches):
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x, env), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = self.ema * return_embedding +\
                               (1 - self.ema) * self.embeddings[env]

            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))

class SagNet(Algorithm):
    """
    Style Agnostic Network
    Algorithm 1 from: https://arxiv.org/abs/1910.11645 
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SagNet, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        # featurizer network
        self.network_f = networks.Featurizer(input_shape, self.hparams)
        # content network
        self.network_c = nn.Linear(self.network_f.n_outputs, num_classes)
        # style network
        self.network_s = nn.Linear(self.network_f.n_outputs, num_classes)

        # # This commented block of code implements something closer to the
        # # original paper, but is specific to ResNet and puts in disadvantage
        # # the other algorithms.
        # resnet_c = networks.Featurizer(input_shape, self.hparams)
        # resnet_s = networks.Featurizer(input_shape, self.hparams)
        # # featurizer network
        # self.network_f = torch.nn.Sequential(
        #         resnet_c.network.conv1,
        #         resnet_c.network.bn1,
        #         resnet_c.network.relu,
        #         resnet_c.network.maxpool,
        #         resnet_c.network.layer1,
        #         resnet_c.network.layer2,
        #         resnet_c.network.layer3)
        # # content network
        # self.network_c = torch.nn.Sequential(
        #         resnet_c.network.layer4,
        #         resnet_c.network.avgpool,
        #         networks.Flatten(),
        #         resnet_c.network.fc)
        # # style network
        # self.network_s = torch.nn.Sequential(
        #         resnet_s.network.layer4,
        #         resnet_s.network.avgpool,
        #         networks.Flatten(),
        #         resnet_s.network.fc)

        def opt(p):
            return torch.optim.Adam(p, lr=hparams["lr"],
                    weight_decay=hparams["weight_decay"])

        self.optimizer_f = opt(self.network_f.parameters())
        self.optimizer_c = opt(self.network_c.parameters())
        self.optimizer_s = opt(self.network_s.parameters())
        self.weight_adv = hparams["sag_w_adv"]

    def forward_c(self, x):
        # learning content network on randomized style
        return self.network_c(self.randomize(self.network_f(x), "style"))

    def forward_s(self, x):
        # learning style network on randomized content
        return self.network_s(self.randomize(self.network_f(x), "content"))
    
    def randomize(self, x, what="style", eps=1e-5):
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1).cuda()

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1) 

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        
        x = (x - mean) / (var + eps).sqrt()
        
        idx_swap = torch.randperm(sizes[0])
        if what == "style":
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, minibatches):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        # learn content
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # learn style 
        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()
       
        # learn adversary
        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {'loss_c': loss_c.item(), 'loss_s': loss_s.item(),
                'loss_adv': loss_adv.item()}

    def predict(self, x):
        return self.network_c(self.network_f(x))


class RSC(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RSC, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.drop_f = (1 - hparams['rsc_f_drop_factor']) * 100
        self.drop_b = (1 - hparams['rsc_b_drop_factor']) * 100
        self.num_classes = num_classes

    def update(self, minibatches):
        # inputs
        all_x = torch.cat([x for x, y in minibatches])
        # labels
        all_y = torch.cat([y for _, y in minibatches])
        # one-hot labels
        all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
        # features
        all_f = self.featurizer(all_x)
        # predictions
        all_p = self.classifier(all_f)

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.cuda()).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = self.classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        all_p_muted_again = self.classifier(all_f * mask)

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
