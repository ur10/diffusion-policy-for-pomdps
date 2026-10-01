import torch
from copy import deepcopy
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from utils import helpers as utl
from policies.rl import RL_ALGORITHMS
import torchkit.pytorch_utils as ptu
from policies.models.recurrent_actor import Actor_RNN


class BCPolicy_RNN_MLP(nn.Module):
    """
    Recurrent Actor RNN
    """

    ARCH = "memory"
    Markov_Actor = False
    Markov_Critic = False

    def __init__(
        self,
        obs_dim,
        action_dim,
        config_seq,
        config_rl,
        # pixel obs
        image_encoder_fn=lambda: None,
        **kwargs
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.clip = config_seq.clip
        self.clip_grad_norm = config_seq.max_norm

        self.algo = RL_ALGORITHMS[config_rl.algo](
            action_dim=action_dim, **config_rl.to_dict()
        )
        assert config_rl.algo in ['bc'], "This policy is only for BC"
 
        # Actor
        self.actor = Actor_RNN(
            obs_dim,
            action_dim,
            config_seq.model,
            config_rl.config_actor,
            self.algo,
            image_encoder=image_encoder_fn(),  # separate weight
        )
        self.actor_optimizer = Adam(self.actor.parameters(), lr=config_rl.actor_lr)

    @torch.no_grad()
    def get_initial_info(self, *args, **kwargs):
        return self.actor.get_initial_info(*args, **kwargs)

    @torch.no_grad()
    def act(
        self,
        prev_internal_state,
        prev_action,
        reward,
        obs,
        deterministic=False,
    ):
        prev_action = prev_action.unsqueeze(0)  # (1, B, dim)
        reward = reward.unsqueeze(0)  # (1, B, 1)
        obs = obs.unsqueeze(0)  # (1, B, dim)

        current_action, current_internal_state = self.actor.act(
            prev_internal_state=prev_internal_state,
            prev_action=prev_action,
            reward=reward,
            obs=obs,
            deterministic=deterministic,
        )

        return current_action, current_internal_state

    def forward(self, actions, rewards, observs, dones, masks):
        """
        For actions a, rewards r, observs o, dones d: (T+1, B, dim)
                where for each t in [0, T], take action a[t], then receive reward r[t], done d[t], and next obs o[t]
                the hidden state h[t](, c[t]) = RNN(h[t-1](, c[t-1]), a[t], r[t], o[t])
                specially, a[0]=r[0]=d[0]=h[0]=c[0]=0.0, o[0] is the initial obs

        The loss is still on the Q value Q(h[t], a[t]) with real actions taken, i.e. t in [1, T]
                based on Masks (T, B, 1)
        """
        assert (
            actions.dim()
            == rewards.dim()
            == dones.dim()
            == observs.dim()
            == masks.dim()
            == 3
        )
        assert (
            actions.shape[0]
            == rewards.shape[0]
            == dones.shape[0]
            == observs.shape[0]
            == masks.shape[0] + 1
        )
        num_valid = torch.clamp(masks.sum(), min=1.0)  # as denominator of loss
        # Trim sequences if configured
        if self.actor.seq_model.trim_during_training:
            max_history_length = self.actor.seq_model.max_history_length
            actions = actions[-max_history_length:]
            rewards = rewards[-max_history_length:]
            observs = observs[-max_history_length:]
            dones = dones[-max_history_length:]
            masks = masks[-(max_history_length - 1):]  # masks is one shorter

        ### Actor loss
        actor_loss_out = self.algo.actor_loss(
            markov_actor=self.Markov_Actor,
            actor=self.actor,
            observs=observs,
            actions=actions,
            rewards=rewards,
        )
        if isinstance(actor_loss_out, tuple):
            policy_loss, rew_loss, obs_loss = actor_loss_out
        else:
            policy_loss, rew_loss, obs_loss = actor_loss_out, 0.0, 0.0
        # masked policy_loss
        policy_loss = (policy_loss * masks).sum() / num_valid
        rew_loss = (rew_loss * masks).sum() / num_valid if isinstance(rew_loss, torch.Tensor) else rew_loss
        obs_loss = (obs_loss * masks).sum() / num_valid if isinstance(obs_loss, torch.Tensor) else obs_loss
        outputs = {"actor_loss": policy_loss.item(), "reward_loss": rew_loss.item() if isinstance(rew_loss, torch.Tensor) else rew_loss,
                   "obs_loss": obs_loss.item() if isinstance(obs_loss, torch.Tensor) else obs_loss}

        self.actor_optimizer.zero_grad()
        (policy_loss + rew_loss + obs_loss).backward()

        if self.clip and self.clip_grad_norm > 0.0:
            grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.clip_grad_norm
            )
            outputs["raw_actor_grad_norm"] = grad_norm.item()

        self.actor_optimizer.step()

        return outputs

    def report_grad_norm(self):
        # may add qf1, policy, etc.
        return {
            "actor_grad_norm": utl.get_grad_norm(self.actor),
            "actor_seq_grad_norm": utl.get_grad_norm(self.actor.seq_model),
        }

    def update(self, batch):
        # all are 3D tensor (T,B,dim)
        actions, rewards, dones = batch["act"], batch["rew"], batch["term"]
        _, batch_size, _ = actions.shape
        if not self.algo.continuous_action:
            # for discrete action space, convert to one-hot vectors
            actions = F.one_hot(
                actions.squeeze(-1).long(), num_classes=self.action_dim
            ).float()  # (T, B, A)

        masks = batch["mask"]
        obs, next_obs = batch["obs"], batch["obs2"]  # (T, B, dim)

        # extend observs, actions, rewards, dones from len = T to len = T+1
        observs = torch.cat((obs[[0]], next_obs), dim=0)  # (T+1, B, dim)
        actions = torch.cat(
            (ptu.zeros((1, batch_size, self.action_dim)).float(), actions), dim=0
        )  # (T+1, B, dim)
        rewards = torch.cat(
            (ptu.zeros((1, batch_size, 1)).float(), rewards), dim=0
        )  # (T+1, B, dim)
        dones = torch.cat(
            (ptu.zeros((1, batch_size, 1)).float(), dones), dim=0
        )  # (T+1, B, dim)

        # import time; t0 = time.time()
        outputs = self.forward(actions, rewards, observs, dones, masks)
        # print("single pass", time.time() - t0)
        return outputs
