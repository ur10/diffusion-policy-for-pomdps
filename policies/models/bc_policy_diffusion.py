import torch
import torch.nn as nn
from torch.optim import Adam

from policies.models.dit.models.diffusion_unet import DiffusionUnetAgent
from policies.models.dit.models.resnet import ResNet
from policies.models.dit.models.vit import (
    vit_base_patch16,
    vit_small_patch16,
    load_vit,
)


def build_features(config_encoder):
    """Instantiate a visual encoder from config."""
    name = config_encoder.name
    if name == "resnet":
        return ResNet(
            size=config_encoder.size,
            norm_cfg=config_encoder.norm_cfg.to_dict(),
            avg_pool=config_encoder.get("avg_pool", True),
            weights=config_encoder.get("weights", None),
            restore_path=config_encoder.get("restore_path", ""),
        )
    elif name == "vit_base":
        model = vit_base_patch16()
        return load_vit(model, config_encoder.get("restore_path", ""))
    elif name == "vit_small":
        model = vit_small_patch16()
        return load_vit(model, config_encoder.get("restore_path", ""))
    else:
        raise NotImplementedError(f"Unknown encoder: {name}")


class BCPolicy_Diffusion(nn.Module):
    """
    BC policy using DiffusionUnetAgent (image-based, non-recurrent).

    Expected batch format for update():
        batch["obs"]  : (B, obs_history, obs_dim)          — proprioceptive history
        batch["imgs"] : dict {"cam0": (B, obs_history, C, H, W), ...}  — image history
        batch["act"]  : (B, ac_chunk, ac_dim) — action chunks
        batch["mask"] : (B, ac_chunk, ac_dim) — validity mask (1=valid, 0=pad)
    """

    ARCH = "markov"
    Markov_Actor = True
    Markov_Critic = True

    def __init__(
        self,
        obs_dim,
        action_dim,
        config_seq,  # unused, kept for interface compatibility
        config_rl,
        image_encoder_fn=lambda: None,  # unused, kept for interface compatibility
        **kwargs,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        cfg = config_rl.config_diffusion
        features = build_features(cfg.encoder)

        obs_history = cfg.get("obs_history", 1)
        self.agent = DiffusionUnetAgent(
            features=features,
            odim=obs_dim,
            n_cams=cfg.n_cams,
            imgs_per_cam=obs_history,
            use_obs=cfg.get("use_obs", False),
            ac_dim=action_dim,
            ac_chunk=cfg.ac_chunk,
            train_diffusion_steps=cfg.train_diffusion_steps,
            eval_diffusion_steps=cfg.eval_diffusion_steps,
            noise_net_kwargs=cfg.noise_net_kwargs.to_dict(),
            obs_history=obs_history,
        )

        self.optimizer = Adam(self.agent.parameters(), lr=config_rl.actor_lr)

    def forward(self, imgs, obs, actions, mask):
        """
        imgs    : dict {"cam0": (B, C, H, W), ...}
        obs     : (B, obs_dim)
        actions : (B, ac_chunk, ac_dim)
        mask    : (B, ac_chunk, ac_dim)
        """
        B, ac_chunk, ac_dim = actions.shape
        ac_flat = actions.reshape(B, ac_chunk * ac_dim)
        mask_flat = mask.reshape(B, ac_chunk * ac_dim)
        return self.agent(imgs, obs, ac_flat, mask_flat)

    def update(self, batch):
        """
        batch must contain keys: obs, imgs, act, mask
        act  : (B, ac_chunk, ac_dim)
        mask : (B, ac_chunk, ac_dim)
        obs  : (B, obs_dim)
        imgs : dict {"cam0": (B, C, H, W), ...}
        """
        obs = batch["obs"]
        imgs = batch["imgs"]
        actions = batch["act"]
        mask = batch["mask"]

        loss = self.forward(imgs, obs, actions, mask)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"diffusion_loss": loss.item()}

    @torch.no_grad()
    def act(self, obs, imgs=None, deterministic=False):
        """
        obs  : (1, obs_dim)
        imgs : dict {"cam0": (1, C, H, W), ...}  — required at inference

        Returns the first action from the predicted chunk: (1, ac_dim)
        """
        action_chunk = self.agent.get_actions(imgs, obs)  # (1, ac_chunk, ac_dim)
        return action_chunk[:, 0, :]  # (1, ac_dim)

    def report_grad_norm(self):
        total_norm = 0.0
        for p in self.agent.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return {"actor_grad_norm": total_norm ** 0.5}
