import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from ..agent import BaseAgent


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TransformerForDiffusion(nn.Module):
    """
    Adapted from long-context-dp's transformer diffusion model for the
    obs-conditioned setting used in this repository.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        horizon,
        n_obs_steps,
        cond_dim,
        n_layer=8,
        n_cond_layers=0,
        n_head=4,
        n_emb=256,
        p_drop_emb=0.0,
        p_drop_attn=0.3,
        causal_attn=True,
    ):
        super().__init__()

        self.horizon = horizon
        self.n_obs_steps = n_obs_steps

        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = nn.Linear(cond_dim, n_emb)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, 1 + n_obs_steps, n_emb))

        if n_cond_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=n_cond_layers
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(n_emb, 4 * n_emb),
                nn.Mish(),
                nn.Linear(4 * n_emb, n_emb),
            )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=n_layer
        )

        if causal_attn:
            tgt_mask = torch.triu(torch.ones(horizon, horizon), diagonal=1)
            tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float("-inf"))
            self.register_buffer("tgt_mask", tgt_mask)

            t, s = torch.meshgrid(
                torch.arange(horizon), torch.arange(1 + n_obs_steps), indexing="ij"
            )
            memory_mask = t >= (s - 1)
            memory_mask = (
                memory_mask.float()
                .masked_fill(memory_mask == 0, float("-inf"))
                .masked_fill(memory_mask == 1, float(0.0))
            )
            self.register_buffer("memory_mask", memory_mask)
        else:
            self.tgt_mask = None
            self.memory_mask = None

        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            SinusoidalPosEmb,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            for name in ("in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight"):
                weight = getattr(module, name, None)
                if weight is not None:
                    nn.init.normal_(weight, mean=0.0, std=0.02)
            for name in ("in_proj_bias", "bias_k", "bias_v"):
                bias = getattr(module, name, None)
                if bias is not None:
                    nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusion):
            nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError(f"Unaccounted module {module}")

    def forward(self, sample, timestep, cond):
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        time_emb = self.time_emb(timesteps).unsqueeze(1)
        cond_obs_emb = self.cond_obs_emb(cond)
        cond_embeddings = torch.cat([time_emb, cond_obs_emb], dim=1)
        cond_embeddings = self.drop(
            cond_embeddings + self.cond_pos_emb[:, : cond_embeddings.shape[1], :]
        )
        memory = self.encoder(cond_embeddings)

        token_embeddings = self.input_emb(sample)
        x = self.drop(token_embeddings + self.pos_emb[:, : token_embeddings.shape[1], :])
        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=self.tgt_mask,
            memory_mask=self.memory_mask,
            tgt_is_causal=self.tgt_mask is not None,
        )

        x = self.ln_f(x)
        return self.head(x)


class LongContextDiffusionTransformerAgent(BaseAgent):
    supports_consistency_sampling = True

    def __init__(
        self,
        features,
        odim,
        n_cams,
        use_obs,
        ac_dim,
        ac_chunk,
        train_diffusion_steps,
        eval_diffusion_steps,
        imgs_per_cam=1,
        dropout=0,
        share_cam_features=False,
        early_fusion=False,
        feat_norm=None,
        token_dim=None,
        noise_net_kwargs=None,
        obs_history=1,
        past_action_pred=True,
        past_steps_reg=-1,
    ):
        self._img_tokens_per_frame = features.n_tokens
        super().__init__(
            odim=odim,
            features=features,
            n_cams=n_cams,
            imgs_per_cam=imgs_per_cam,
            use_obs=use_obs,
            share_cam_features=share_cam_features,
            early_fusion=early_fusion,
            dropout=dropout,
            feat_norm=feat_norm,
            token_dim=token_dim,
            obs_history=obs_history,
        )

        if early_fusion:
            raise NotImplementedError("PTP agent does not support early_fusion.")

        noise_net_kwargs = dict(noise_net_kwargs or {})
        self._ac_dim = ac_dim
        self._ac_chunk = ac_chunk
        self._obs_history = obs_history
        self._horizon = obs_history + ac_chunk - 1
        self._past_action_pred = past_action_pred
        self._past_steps_reg = past_steps_reg

        cond_dim = n_cams * self._img_tokens_per_frame * self.token_dim
        if self._obs_strat == "add_token":
            cond_dim += self.token_dim

        self.model = TransformerForDiffusion(
            input_dim=ac_dim,
            output_dim=ac_dim,
            horizon=self._horizon,
            n_obs_steps=obs_history,
            cond_dim=cond_dim,
            **noise_net_kwargs,
        )

        assert eval_diffusion_steps <= train_diffusion_steps
        self._train_diffusion_steps = train_diffusion_steps
        self._eval_diffusion_steps = eval_diffusion_steps
        self.diffusion_schedule = DDIMScheduler(
            num_train_timesteps=train_diffusion_steps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        )

    def _encode_condition(self, imgs, obs):
        if obs.ndim == 2:
            obs = obs[:, None, :]

        tokens = self.tokenize_obs(imgs, obs)
        B, _, D = tokens.shape

        if self._obs_strat == "add_token":
            obs_tokens = tokens[:, -self._obs_history :]
            img_tokens = tokens[:, : -self._obs_history]
        else:
            obs_tokens = None
            img_tokens = tokens

        img_tokens = img_tokens.reshape(
            B,
            self.n_cams,
            self._obs_history,
            self._img_tokens_per_frame,
            D,
        )
        img_tokens = img_tokens.permute(0, 2, 1, 3, 4).reshape(B, self._obs_history, -1)

        if obs_tokens is not None:
            img_tokens = torch.cat([img_tokens, obs_tokens], dim=-1)
        return img_tokens

    def forward(self, imgs, obs, action_seq, mask_seq):
        cond = self._encode_condition(imgs, obs)
        noise = torch.randn_like(action_seq)
        timesteps = torch.randint(
            0,
            self._train_diffusion_steps,
            (action_seq.shape[0],),
            device=action_seq.device,
        ).long()
        noisy_actions = self.diffusion_schedule.add_noise(action_seq, noise, timesteps)
        pred = self.model(noisy_actions, timesteps, cond)
        target = noise
        loss_mask = mask_seq

        start = 0
        if not self._past_action_pred:
            start = self._obs_history - 1
        if self._past_steps_reg != -1:
            start = max(start, self._obs_history - self._past_steps_reg - 1)

        pred = pred[:, start:]
        target = target[:, start:]
        loss_mask = loss_mask[:, start:]

        loss = F.mse_loss(pred, target, reduction="none")
        loss_mask = loss_mask.to(loss.dtype)
        denom = loss_mask.sum(dim=(1, 2)).clamp_min(1.0)
        loss = (loss * loss_mask).sum(dim=(1, 2)) / denom
        return loss.mean()

    def sample_trajectories(self, imgs, obs, n_samples=1, n_steps=None):
        cond = self._encode_condition(imgs, obs)
        B = cond.shape[0]
        device = cond.device

        cond = cond.repeat_interleave(n_samples, dim=0)
        traj = torch.randn(
            cond.shape[0], self._horizon, self._ac_dim, device=device, dtype=cond.dtype
        )

        eval_steps = self._eval_diffusion_steps if n_steps is None else n_steps
        self.diffusion_schedule.set_timesteps(eval_steps, device=device)

        for timestep in self.diffusion_schedule.timesteps:
            batched_timestep = timestep.expand(cond.shape[0]).to(device)
            model_output = self.model(traj, batched_timestep, cond)
            traj = self.diffusion_schedule.step(
                model_output, timestep, traj
            ).prev_sample

        return traj.reshape(n_samples, B, self._horizon, self._ac_dim)

    def get_actions(
        self,
        imgs,
        obs,
        n_steps=None,
        past_actions=None,
        n_samples=None,
        return_full=False,
    ):
        sample_count = max(int(n_samples or 1), 1)
        trajs = self.sample_trajectories(imgs, obs, n_samples=sample_count, n_steps=n_steps)

        if (
            sample_count > 1
            and past_actions is not None
            and self._obs_history > 1
            and past_actions.shape[1] >= self._obs_history - 1
        ):
            past_actions = past_actions[:, -(self._obs_history - 1) :].to(trajs.device)
            pred_past = trajs[:, :, : self._obs_history - 1]
            mses = F.mse_loss(
                pred_past,
                past_actions.unsqueeze(0).expand_as(pred_past),
                reduction="none",
            )
            mses = mses.mean(dim=(2, 3))
            best_idx = mses.argmin(dim=0)
            batch_idx = torch.arange(trajs.shape[1], device=trajs.device)
            traj = trajs[best_idx, batch_idx]
        else:
            traj = trajs[0]

        if return_full:
            return traj

        start = self._obs_history - 1
        end = start + self._ac_chunk
        return traj[:, start:end]

    @property
    def ac_chunk(self):
        return self._ac_chunk

    @property
    def ac_dim(self):
        return self._ac_dim
