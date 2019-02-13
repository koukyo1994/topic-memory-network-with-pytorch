import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from core.nn.model import NeuralTopicModel, TopicMemoryNetwork


class NTMLoss:
    def __init__(self, kl_strength):
        self.kl_strength = kl_strength
        self.nnl = nn.CrossEntropyLoss(reduction="sum")

    def __call__(self, p_x_given_h, x_true, z_mean, z_log_var):
        NNL = self.nnl(p_x_given_h, x_true)
        KLD = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        return self.kl_strength * KLD + NNL

    def update(self, kl_strength):
        self.kl_strength = kl_strength


class Trainer:
    def __init__(self,
                 logger,
                 ntm_params={},
                 tmn_params={},
                 kl_growing_epoch=0):
        self.ntm_params = ntm_params
        self.ntm = NeuralTopicModel(**ntm_params)
        self.ntm.cuda()

        self.tmn_params = tmn_params
        self.tmn = TopicMemoryNetwork(**tmn_params)
        self.tmn.cuda()

        self.ntm_optimizer = optim.SparseAdam(self.ntm.parameters())
        self.tmn_optimizer = optim.SparseAdam(self.tmn.parameters())

        self.kl_strength = 1.0
        self.ntm_loss_fn = NTMLoss(self.kl_strength)

        self.logger = logger

        self.optimize_ntm = True
        self.first_optimize_ntm = True
        self.min_bound_ntm = np.inf
        self.min_bound_cls = -np.inf
        self.epoch_since_improvement = 0
        self.epoch_since_improvement_global = 0

        self.psudo_indices = np.expand_dims(
            np.arange(self.ntm_params.get("n_topics")), axis=0)
        self.kl_growing_epoch = kl_growing_epoch

        self.max_epochs = 0
        self.current_epoch = 0

    def _train_ntm(self, train_loader):
        num_batches = len(train_loader)
        avg_loss = 0.
        kl_base = float(self.kl_growing_epoch * num_batches)
        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch
            self.ntm.train()
            self.logger.info(f"Epoch {epoch+1}/{self.max_epochs} training NTM")
            for i, (_, bow, label) in tqdm(enumerate(train_loader)):
                if epoch + 1 < self.kl_growing_epoch:
                    self.kl_strength = np.float32(
                        ((epoch + 1) * num_batches + i) / kl_base)
                    self.ntm_loss_fn.update(self.kl_strength)
                else:
                    self.kl_strength = 1.0
                    self.ntm_loss_fn.update(self.kl_strength)
                _, p_x_given_h, z_mean, z_log_var = self.ntm(bow)
                loss = self.ntm_loss_fn(p_x_given_h, bow, z_mean, z_log_var)
                self.ntm_optimizer.zero_grad()
                loss.backward()
                self.ntm_optimizer.step()
                avg_loss += loss.item() / num_batches
            self.logger.info(f"NTM train loss: {avg_loss:.3f}")

    def train(self, train_loader, max_epochs):
        self.max_epochs = max_epochs
        if self.optimize_ntm:
            self._train_ntm(train_loader)
