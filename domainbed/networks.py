# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from domainbed.lib import misc
from domainbed.lib import wide_resnet

from einops import rearrange

class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class SqueezeLastTwo(nn.Module):
    """A module which squeezes the last two dimensions, ordinary squeeze can be a problem for batch size 1"""
    def __init__(self):
        super(SqueezeLastTwo, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], x.shape[1])


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'],hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x

class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams['resnet18']:
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(pretrained=True)
            self.n_outputs = 2048

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])
        self.flattenLayer = nn.Flatten()

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        x = self.network.conv1(x)
        x = self.network.bn1(x)
        x = self.network.relu(x)
        x = self.network.maxpool(x)
        x = self.network.layer1(x)
        x = self.network.layer2(x)
        x = self.network.layer3(x)
        x = self.network.layer4(x)
        x = self.network.avgpool(x)
        x = self.flattenLayer(x)
        x = self.dropout(x)
        return x

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.squeezeLastTwo = SqueezeLastTwo()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = self.squeezeLastTwo(x)
        return x

class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


class CosineClassifier(nn.Module):
    def __init__(self, classes, channels=512):
        super().__init__()
        self.channels = channels
        self.cls = nn.Conv2d(channels, classes, 1, bias=False)
        self.scaler = 10.

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        return self.scaler * F.conv2d(x, F.normalize(self.cls.weight, dim=1, p=2))


class PPLayer(nn.Module):
    def __init__(self, support_set_shape, use_eval_cache, support_set_shape_eval, use_attention):
        super(PPLayer, self).__init__()
        self.support_set_shape = support_set_shape
        self.num_domains = support_set_shape[0]
        self.num_classes = support_set_shape[1]
        self.num_images_per_class = support_set_shape[2]
        self.num_images = self.num_domains *  self.num_classes * self.num_images_per_class
        self.use_attention = use_attention

        self.cache =  nn.Parameter(torch.rand(support_set_shape), requires_grad=False) 
        self.cache_mask = nn.Parameter(torch.zeros(self.num_domains, self.num_classes, self.num_images_per_class), requires_grad=False)

        if use_attention:
            self.dim_key = 128
            self.dim_value = 128
            self.to_qk = nn.Conv2d(512, self.dim_key, 1, bias = False)
            self.to_v = nn.Conv2d(512, self.dim_value, 1, bias = False)

        if use_eval_cache:
            self.num_images_per_class_eval = support_set_shape_eval[2]
            self.num_images_eval = self.num_domains *  self.num_classes * self.num_images_per_class_eval
            self.cache_eval =  torch.FloatTensor(torch.rand(support_set_shape_eval)).cpu()
            self.cache_mask_eval = torch.FloatTensor(torch.zeros(self.num_domains, self.num_classes, self.num_images_per_class_eval)).cpu()

    def forward(self, x, featurizer):
        prototypes = self.get_prototypes(featurizer, self.cache)
        if self.use_attention:
            """
            dimensions names:
        
            b - batch size
            k - num classes
            d - num domains
            n - num images in a support class
            c - num channels
            h, i - height
            w, j - width
            """
            query_q, query_v = self.to_qk(x), self.to_v(x)
            reshaped_prots = prototypes.view(self.num_images, featurizer.n_outputs, 7, 7)
            prot_k, prot_v = self.to_qk(reshaped_prots), self.to_v(reshaped_prots)
            prot_k, prot_v = map(lambda t: t.view(self.num_domains, self.num_classes, self.num_images_per_class, self.dim_key, 7, 7), (prot_k, prot_v))
            similarity_per_location = self.prototype_similarities(query_q, prot_k, self.num_images)
            sim = rearrange(similarity_per_location, 'b (d k n) (i j) h w -> b d k h w (n i j)', d = self.num_domains, k = self.num_classes, i = 7, j = 7)
            attn = (sim / math.sqrt(self.dim_key)).softmax(dim = -1)
            attn = rearrange(attn, 'b d k h w (n i j) -> b d k n h w i j', i = 7, j = 7)
            out = torch.einsum('b d k n h w i j, d k n c i j -> b d k c h w', attn, prot_v)
            out = out.reshape(query_q.shape[0], self.num_domains, self.num_classes, -1)
            query_v = rearrange(query_v, 'b c h w -> b () () (c h w)')
            euclidean_distance = ((query_v - out) ** 2).sum(dim = -1) / (x.shape[-2] * x.shape[-1])
            return -euclidean_distance # has shape [B, num_domains, num_classes]

        else:
            prototypes = prototypes.mean(-1).mean(-1).unsqueeze(-1).unsqueeze(-1)                                                           # AVERAGING
            x = x.mean(-1).mean(-1).unsqueeze(-1).unsqueeze(-1)                                                                             # AVERAGING
            similarity_per_location = self.prototype_similarities(x, prototypes, self.num_images)
            pooled_similarity = similarity_per_location.max(2)[0]                                                                           # Maximum similarity per image per location [BS, num_images, 7, 7]
            proto_scores = F.max_pool2d(pooled_similarity, kernel_size=(pooled_similarity.size()[2], pooled_similarity.size()[3]))          # Maximum similarity per image [B, num_images, 1, 1]
            prototype_activations = proto_scores.view(-1, self.num_domains, self.num_classes, self.num_images_per_class).max(-1)[0]         # has shape [B, num_domains, num_classes]
            return prototype_activations

    def forward_eval(self, x, featurizer):
        with torch.no_grad():
            batch_size = 25
            output = []
            x = x.mean(-1).mean(-1).unsqueeze(-1).unsqueeze(-1)                                                                 # AVERAGING
            for batch in range(math.ceil(self.num_images_per_class_eval / batch_size)):
                cache_subset = self.cache_eval[:, :, batch * batch_size : (batch+1) * batch_size, :, :, :].cuda()
                num_images_subset = cache_subset.shape[0] * cache_subset.shape[1] * cache_subset.shape[2]
                prototype_subset = self.get_prototypes(featurizer, cache_subset)
                prototype_subset = prototype_subset.mean(-1).mean(-1).unsqueeze(-1).unsqueeze(-1)                               # AVERAGING
                similarity_per_location = self.prototype_similarities(x, prototype_subset, num_images_subset)
                pooled_similarity = similarity_per_location.max(2)[0]
                proto_scores = F.max_pool2d(pooled_similarity, kernel_size=(pooled_similarity.size()[2], pooled_similarity.size()[3]))
                prototype_activations = proto_scores.view(x.shape[0], self.num_domains, self.num_classes, -1)
                output.append(prototype_activations)
            proto_scores = torch.cat(output, dim=3).view(x.shape[0], -1)
            prototype_activations = proto_scores.view(-1, self.num_domains, self.num_classes, self.num_images_per_class_eval).max(-1)[0]
            return prototype_activations



    def input_features(self, x):
        """
        Input features to the prototype layer, the original ProtoPNet uses some add_on_layers after the feature extractor
        """
        return x

    def get_prototypes(self, featurizer, cache):
        prototypes = featurizer(cache.view(-1, self.support_set_shape[3], self.support_set_shape[4], self.support_set_shape[5])).clone().detach()
        prototypes = prototypes.view(self.num_domains, self.num_classes, -1, featurizer.n_outputs, 7, 7)
        return prototypes

    def _dot_similarity(self, x, prototypes, num_images):
        reshaped_prots = prototypes.permute(0, 1, 2, 4, 5, 3).reshape(-1, prototypes.shape[3], 1, 1)
        similarity = F.conv2d(input=x, weight=reshaped_prots)
        similarity_per_location = similarity.view(-1, num_images, prototypes.shape[4] * prototypes.shape[5], x.shape[2], x.shape[3])
        return similarity_per_location

    def prototype_similarities(self, x, prototypes, num_images):
        """
        x are the features from the feature extractor, we call input_features to possibly pass additional layers in between
        """
        features = self.input_features(x)
        distances = self._dot_similarity(features, prototypes, num_images)
        return distances



def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], 128, hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    elif input_shape[1:3] == (224, 224):
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError
