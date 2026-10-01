import torch
import torch.nn as nn
from torch.optim import AdamW

from diffusers.optimization import get_scheduler

from policies.models.ema import EMAModel
from policies.models.dit.models.ldp_transformer import LongContextDiffusionTransformerAgent
from policies.models.dit.models.resnet import ResNet
from policies.models.dit.models.vit import (
    load_vit,
    vit_base_patch16,
    vit_small_patch16,
)


def build_features(config_encoder):
    name = config_encoder.name
    if name == "resnet":
        return ResNet(
            size=config_encoder.size,
            norm_cfg=config_encoder.norm_cfg.to_dict(),
            avg_pool=config_encoder.get("avg_pool", True),
            weights=config_encoder.get("weights", None),
            restore_path=config_encoder.get("restore_path", ""),
        )
    if name == "vit_base":
        model = vit_base_patch16()
        return load_vit(model, config_encoder.get("restore_path", ""))
    if name == "vit_small":
        model = vit_small_patch16()
        return load_vit(model, config_encoder.get("restore_path", ""))
    raise NotImplementedError(f"Unknown encoder: {name}")


class BCPolicy_LDP(nn.Module):
    ARCH = "markov"
    Markov_Actor = True
    Markov_Critic = True

    def __init__(
        self,
        obs_dim,
        action_dim,
        config_seq,
        config_rl,
        image_encoder_fn=lambda: None,
        **kwargs,
    ):
        super().__init__()
        del config_seq, image_encoder_fn, kwargs

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        cfg = config_rl.config_diffusion
        features = build_features(cfg.encoder)

        self.agent = LongContextDiffusionTransformerAgent(
            features=features,
            odim=obs_dim,
            n_cams=cfg.n_cams,
            imgs_per_cam=cfg.get("obs_history", 1),
            use_obs=cfg.get("use_obs", False),
            ac_dim=action_dim,
            ac_chunk=cfg.ac_chunk,
            train_diffusion_steps=cfg.train_diffusion_steps,
            eval_diffusion_steps=cfg.eval_diffusion_steps,
            noise_net_kwargs=cfg.noise_net_kwargs.to_dict(),
            obs_history=cfg.get("obs_history", 1),
            past_action_pred=cfg.get("past_action_pred", True),
            past_steps_reg=cfg.get("past_steps_reg", -1),
        )

        # Split params: encoder (visual_features) gets a small weight decay
        # to mirror LDP's obs_encoder_weight_decay; everything else gets the
        # transformer_weight_decay value.
        encoder_decay = float(config_rl.get("encoder_weight_decay", 1e-6))
        transformer_decay = float(config_rl.get("weight_decay", 1e-3))
        encoder_params, other_params = [], []
        for name, p in self.agent.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("visual_features"):
                encoder_params.append(p)
            else:
                other_params.append(p)

        self.optimizer = AdamW(
            [
                {"params": encoder_params, "weight_decay": encoder_decay},
                {"params": other_params, "weight_decay": transformer_decay},
            ],
            lr=config_rl.actor_lr,
            betas=tuple(config_rl.get("betas", (0.9, 0.95))),
        )

        # Cosine LR schedule with warmup, matching LDP's
        # lr_scheduler=cosine + lr_warmup_steps=1000.
        train_steps = int(config_rl.get("train_steps", 30_000))
        warmup_steps = int(config_rl.get("lr_warmup_steps", 1000))
        self.lr_scheduler = get_scheduler(
            name=config_rl.get("lr_scheduler", "cosine"),
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=train_steps,
        )

        # Compile hot sub-modules before EMA so the shadow gets compiled copies too.
        self.agent.model = torch.compile(self.agent.model, mode="default")
        if isinstance(self.agent.visual_features, nn.ModuleList):
            for i, feat in enumerate(self.agent.visual_features):
                self.agent.visual_features[i] = torch.compile(feat, mode="default")
        else:
            self.agent.visual_features = torch.compile(self.agent.visual_features, mode="default")

        # EMA tracker — keeps a deep-copied shadow model whose weights are
        # the exponentially-averaged version of self.agent. Mirrors the
        # `diffusion_policy.model.diffusion.ema_model.EMAModel` used by LDP.
        self.use_ema = bool(config_rl.get("use_ema", True))
        if self.use_ema:
            self.ema = EMAModel(
                model=self.agent,
                power=float(config_rl.get("ema_power", 0.75)),
                inv_gamma=float(config_rl.get("ema_inv_gamma", 1.0)),
                min_value=float(config_rl.get("ema_min_value", 0.0)),
                max_value=float(config_rl.get("ema_max_value", 0.9999)),
                update_after_step=int(config_rl.get("ema_update_after_step", 0)),
            )
            # Register the EMA shadow as a submodule so its parameters land in
            # state_dict() and survive across checkpoint save/load. Without
            # this, eval would build a fresh BCPolicy_LDP, the shadow would be
            # initialized from random fresh weights via copy.deepcopy, and the
            # eval-time swap would feed those random weights to the rollout.
            self.ema_shadow = self.ema.averaged_model
        else:
            self.ema = None

        self.max_grad_norm = float(config_rl.get("max_grad_norm", 1.0))

        # During eval we expose the EMA shadow as `self.agent` (the rollout
        # in learner.evaluate uses `self.agent.agent`). Stash the live model
        # in a plain list so nn.Module's __setattr__ does not double-register
        # it as a submodule.
        self._live_agent_ref = [self.agent]

    def load_state_dict(self, state_dict, strict=True):
        """Tolerant load: pre-fix checkpoints saved without ema_shadow.* keys
        will copy the live model's weights into the shadow as a fallback so
        the eval-time EMA swap doesn't expose random fresh-init weights.
        """
        has_shadow_keys = any(k.startswith("ema_shadow.") for k in state_dict)
        if self.use_ema and not has_shadow_keys:
            # Fall back: pretend ema_shadow weights == live agent weights.
            # Mirrors what would happen if EMA decay had been 0.0 (shadow == live).
            for k, v in list(state_dict.items()):
                if k.startswith("agent."):
                    state_dict["ema_shadow." + k[len("agent."):]] = v
            print(
                "[BCPolicy_LDP] load: ema_shadow.* keys missing in checkpoint — "
                "filling shadow with live-model weights (use_ema=False to skip)."
            )
        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, imgs, obs, action_seq, mask_seq):
        return self.agent(imgs, obs, action_seq, mask_seq)

    def update(self, batch):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = self.forward(
                batch["imgs"],
                batch["obs"],
                batch["act_seq"],
                batch["mask_seq"],
            )

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.agent.parameters(), self.max_grad_norm
            )
        self.optimizer.step()
        self.lr_scheduler.step()
        if self.ema is not None:
            self.ema.step(self._live_agent_ref[0])

        return {
            "diffusion_loss": loss.item(),
            "lr": self.lr_scheduler.get_last_lr()[0],
        }

    def eval(self):
        super().eval()
        if self.ema is not None:
            # Route inference through the EMA shadow model.
            self.agent = self.ema.averaged_model
        return self

    def train(self, mode=True):
        if mode and self.ema is not None:
            self.agent = self._live_agent_ref[0]
        super().train(mode)
        return self

    @torch.no_grad()
    def act(self, obs, imgs=None, deterministic=False):
        del deterministic
        action_chunk = self.agent.get_actions(imgs, obs)
        return action_chunk[:, 0, :]

    def report_grad_norm(self):
        total_norm = 0.0
        for p in self.agent.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return {"actor_grad_norm": total_norm ** 0.5}
