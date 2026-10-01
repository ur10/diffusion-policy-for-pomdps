import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

JEPA_REPO_PATH = "models/jepa"
sys.path.insert(0, JEPA_REPO_PATH)
import src.models.vision_transformer as vit

from policies.models.dit.models.diffusion_dit import DiffusionTransformerAgent

JEPA_CHECKPOINT = "models/jepa/checkpoints/vitl16.pth.tar"


class VJEPAEncoder(nn.Module):
    """
    Frozen V-JEPA ViT-L video encoder.

    Accepts a full video clip [B, 3, T, H, W] (values in [0, 1]) and returns
    spatially mean-pooled temporal tokens [B, T_enc // 2, 1024].

    T_enc = max(obs_history, 2) rounded up to the nearest even number, which
    satisfies the tubelet_size=2 requirement. Frame counts other than 16 are
    handled by V-JEPA's trilinear positional-embedding interpolation.
    """

    def __init__(self, obs_history, checkpoint_path=JEPA_CHECKPOINT, finetune=False):
        super().__init__()

        T = max(2, obs_history)
        if T % 2 != 0:
            T += 1
        self._T = T
        self._finetune = finetune

        self._encoder = vit.vit_large(
            img_size=224,
            patch_size=16,
            num_frames=T,
            tubelet_size=2,
            uniform_power=True,
            use_sdpa=True,
            use_SiLU=False,
            tight_SiLU=False,
        )
        print(f"Loading V-JEPA checkpoint from {checkpoint_path} ...")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        sd = {k.replace('module.', '').replace('backbone.', ''): v
              for k, v in ckpt['target_encoder'].items()}
        self._encoder.load_state_dict(sd, strict=False)
        print("V-JEPA checkpoint loaded.")

        if not finetune:
            for p in self._encoder.parameters():
                p.requires_grad_(False)
            self._encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if not self._finetune:
            self._encoder.eval()
        return self

    def forward(self, x):
        """
        x : [B, 3, T_in, H, W]  float32, already ImageNet-normalized
        Returns [B, T_enc // 2, 1024]  (spatially mean-pooled temporal tokens)
        """
        _, C, T_in, H, W = x.shape

        # Pad to T_enc by repeating the last frame if needed.
        if T_in < self._T:
            pad = x[:, :, -1:].expand(-1, -1, self._T - T_in, -1, -1)
            x = torch.cat([x, pad], dim=2)

        # Images arrive pre-normalized (RoboSuiteBuffer applies the "preproc"
        # transform which already applies ImageNet mean/std).  Do NOT normalize
        # again here — double normalization produces values outside V-JEPA's
        # expected input range and destroys the extracted features.

        if self._finetune:
            tokens = self._encoder(x)                        # [B, (T//2)*196, 1024]
            tokens = F.layer_norm(tokens, (1024,))
        else:
            with torch.no_grad():
                tokens = self._encoder(x)                    # [B, (T//2)*196, 1024]
                tokens = F.layer_norm(tokens, (1024,))

        B, _, D = tokens.shape
        tokens = tokens.reshape(B, self._T // 2, 196, D).mean(dim=2)  # [B, T//2, 1024]
        return tokens

    @property
    def embed_dim(self):
        return 1024

    @property
    def n_tokens(self):
        return self._T // 2  # one temporal token per tubelet pair


class JEPADiffusionTransformerAgent(DiffusionTransformerAgent):
    """
    DiffusionTransformerAgent with V-JEPA temporal visual encoding.

    Overrides embed() so the full [B, T, C, H, W] obs_history is passed as a
    single video clip [B, C, T, H, W] to the frozen JEPA encoder, rather than
    processing frames one by one.
    """

    def embed(self, imgs):
        """
        imgs : dict {"cam0": [B, T, 3, H, W], ...}
        Returns [B, n_cams * (T_enc//2) * 196, 1024]
        """
        cam_features = (
            [self.visual_features] * self._n_cams
            if self._share_cam_features
            else list(self.visual_features)
        )
        clips = []
        for i, net in enumerate(cam_features):
            im = imgs[f"cam{i}"]            # [B, T, 3, H, W]
            im = im.permute(0, 2, 1, 3, 4)  # [B, 3, T, H, W]
            clips.append(net(im))           # [B, (T_enc//2)*196, 1024]
        return torch.cat(clips, dim=1)

    def make_pad_mask(self, obs_valid_len, B, device):
        # JEPA encodes the full clip jointly; frame-level padding masks are not
        # applicable here.
        return None


class BCPolicy_JEPA(nn.Module):
    """
    Diffusion-Transformer BC policy with a frozen V-JEPA video encoder.

    The full obs_history frames are treated as a video clip and encoded once by
    V-JEPA (ViT-L, 305 M params, frozen), producing spatio-temporal patch tokens
    that condition the DiT diffusion denoiser.  Only the denoiser is trained.

    Expected batch format for update():
        batch["obs"]  : (B, obs_history, obs_dim)
        batch["imgs"] : dict {"cam0": (B, obs_history, C, H, W)}
        batch["act"]  : (B, ac_chunk, ac_dim)
        batch["mask"] : (B, ac_chunk, ac_dim)
    """

    ARCH = "markov"
    Markov_Actor = True
    Markov_Critic = True

    def __init__(
        self,
        obs_dim,
        action_dim,
        config_seq,   # unused, kept for interface compatibility
        config_rl,
        image_encoder_fn=lambda: None,  # unused, kept for interface compatibility
        **kwargs,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        cfg = config_rl.config_diffusion
        obs_history = cfg.get("obs_history", 1)

        finetune_jepa = cfg.get("finetune_jepa", False)
        encoder = VJEPAEncoder(
            obs_history=obs_history,
            checkpoint_path=cfg.get("jepa_checkpoint", JEPA_CHECKPOINT),
            finetune=finetune_jepa,
        )

        # cond_dim tells noise_net to project JEPA tokens (1024) → hidden_dim (512)
        # once in forward_enc before cross-attention in each decoder block.
        noise_net_kwargs = {**cfg.noise_net_kwargs.to_dict(), "cond_dim": encoder.embed_dim}
        self.agent = JEPADiffusionTransformerAgent(
            features=encoder,
            odim=obs_dim,
            n_cams=cfg.n_cams,
            imgs_per_cam=1,
            use_obs=cfg.get("use_obs", False),
            ac_dim=action_dim,
            ac_chunk=cfg.ac_chunk,
            train_diffusion_steps=cfg.train_diffusion_steps,
            eval_diffusion_steps=cfg.eval_diffusion_steps,
            noise_net_kwargs=noise_net_kwargs,
            obs_history=obs_history,
        )

        if finetune_jepa:
            # Two param groups: JEPA encoder at a lower LR to avoid destroying pretrained representations.
            encoder_ids = {id(p) for p in encoder.parameters()}
            jepa_params = [p for p in self.agent.parameters() if id(p) in encoder_ids]
            denoiser_params = [p for p in self.agent.parameters() if id(p) not in encoder_ids]
            jepa_lr = cfg.get("jepa_lr", config_rl.actor_lr * 0.1)
            self.optimizer = Adam([
                {"params": denoiser_params, "lr": config_rl.actor_lr},
                {"params": jepa_params, "lr": jepa_lr},
            ])
        else:
            trainable = [p for p in self.agent.parameters() if p.requires_grad]
            self.optimizer = Adam(trainable, lr=config_rl.actor_lr)

    def forward(self, imgs, obs, actions, mask):
        B, ac_chunk, ac_dim = actions.shape
        ac_flat = actions.reshape(B, ac_chunk * ac_dim)
        mask_flat = mask.reshape(B, ac_chunk * ac_dim)
        return self.agent(imgs, obs, ac_flat, mask_flat)

    def update(self, batch):
        self.optimizer.zero_grad()
        loss = self.forward(batch["imgs"], batch["obs"], batch["act"], batch["mask"])
        loss.backward()
        self.optimizer.step()
        return {"diffusion_loss": loss.item()}

    @torch.no_grad()
    def act(self, obs, imgs=None, deterministic=False):
        """
        obs  : (1, obs_dim)
        imgs : dict {"cam0": (1, T, C, H, W)}

        Returns the first action of the predicted chunk: (1, ac_dim)
        """
        action_chunk = self.agent.get_actions(imgs, obs)  # (1, ac_chunk, ac_dim)
        return action_chunk[:, 0, :]

    def report_grad_norm(self):
        total_norm = 0.0
        for p in self.agent.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return {"actor_grad_norm": total_norm ** 0.5}
