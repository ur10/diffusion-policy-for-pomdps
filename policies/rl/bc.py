import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from .base import RLAlgorithmBase
from policies.models.actor import CategoricalPolicy
from torchkit.networks import FlattenMlp
import torchkit.pytorch_utils as ptu


class BC(RLAlgorithmBase):
    name = "bc"
    continuous_action = False

    @staticmethod
    def build_actor(input_size, action_dim, hidden_sizes, **kwargs):
        return CategoricalPolicy(
            obs_dim=input_size,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            **kwargs,
        )

    def select_action(self, actor, observ, deterministic: bool):
        return actor(observ, deterministic, return_log_prob=False)[0]

    @staticmethod
    def forward_actor(actor, observ):
        _, probs, log_probs = actor(observ, return_log_prob=True)
        return probs, log_probs  # (T+1, B, dim), (T+1, B, dim)

    def actor_loss(
        self,
        markov_actor: bool,
        actor,
        observs,
        actions=None,
        rewards=None,
        **kwargs,
    ):
        if markov_actor:
            _, log_probs = self.forward_actor(actor, observs)
        else:
            out = actor(
                prev_actions=actions, rewards=rewards, observs=observs
            )  # (T+1, B, A)
            if len(out) == 4:
                _, log_probs, rew_pred, obs_pred = out
            else:
                _, log_probs = out
                rew_pred, obs_pred = None, None

        policy_loss = -(actions[1:] * log_probs[:-1]).sum(axis=-1, keepdims=True)

        if not markov_actor:
            rew_loss = ((rew_pred[:-1] - rewards[1:]) ** 2) if rew_pred is not None else 0.0
            obs_loss = ((obs_pred[:-1] - observs[1:]) ** 2).sum(axis=-1, keepdims=True) if obs_pred is not None else 0.0
            return policy_loss, rew_loss, obs_loss

        return policy_loss