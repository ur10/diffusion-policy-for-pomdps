import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from transformers import VideoMAEModel

from policies.models.dit.models.diffusion_dit import DiffusionTransformerAgent
from policies.models.dit.models.diffusion_ditx import DiTX
from policies.models.ema import EMAModel


class DiTXNoiseNetAdapter(nn.Module):
    """
    Wraps DiTX as a drop-in replacement for _DiTNoiseNet.

    Bridges _DiTNoiseNet's constructor interface (ac_dim, ac_chunk, hidden_dim, …)
    to DiTX's parameter names, and exposes the same forward / forward_enc /
    forward_dec API so DiffusionTransformerAgent works unchanged.
    """

    def __init__(
        self,
        ac_dim: int,
        ac_chunk: int,
        visual_cond_len: int,
        cond_dim: int,
        hidden_dim: int = 768,
        num_blocks: int = 12,
        nhead: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        time_dim: int = 256,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        pre_norm_modality: bool = True,
    ):
        super().__init__()
        self.model = DiTX(
            input_dim=ac_dim,
            output_dim=ac_dim,
            horizon=ac_chunk,
            n_obs_steps=1,
            cond_dim=cond_dim,
            visual_cond_len=visual_cond_len,
            diffusion_timestep_embed_dim=time_dim,
            n_layer=num_blocks,
            n_head=nhead,
            n_emb=hidden_dim,
            mlp_ratio=mlp_ratio,
            p_drop_attn=dropout,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            pre_norm_modality=pre_norm_modality,
        )

    def forward_enc(self, vis_cond, key_padding_mask=None):
        return self.model.forward_enc(vis_cond, key_padding_mask)

    def forward_dec(self, noise_actions, timestep, context_c):
        return self.model.forward_dec(noise_actions, timestep, context_c)

    def forward(self, noise_actions, timestep, vis_cond, enc_cache=None):
        if enc_cache is None:
            enc_cache = self.forward_enc(vis_cond)
        return enc_cache, self.forward_dec(noise_actions, timestep, enc_cache)

VIDEOMAE_MODEL = "MCG-NJU/videomae-base-finetuned-kinetics"


class VideoMAEEncoder(nn.Module):
    """
    V-VideoMAE video encoder (HuggingFace).

    Accepts a full video clip [B, 3, T_in, H, W] (ImageNet-normalized) and
    returns spatially mean-pooled temporal tokens [B, T_model//2, hidden_dim].

    Input is always padded / truncated to the model's native frame count
    (num_frames from config, typically 16).  Unlike V-JEPA there is no
    positional-embedding interpolation for arbitrary T, so we fix T to the
    checkpoint's trained value.
    """

    def __init__(self, model_name=VIDEOMAE_MODEL, finetune=False, freeze_encoder_blocks=0,
                 use_lora=False, lora_r=8, lora_alpha=16):
        super().__init__()
        self._finetune = finetune or use_lora

        print(f"Loading VideoMAE model from {model_name} ...")
        self._encoder = VideoMAEModel.from_pretrained(model_name, attn_implementation="sdpa")
        print("VideoMAE model loaded.")

        cfg = self._encoder.config
        self._T = cfg.num_frames                              # e.g. 16
        self._hidden_dim = cfg.hidden_size                    # 768 (base) / 1024 (large)
        patches_per_side = cfg.image_size // cfg.patch_size   # 14 for 224/16
        self._spatial_patches = patches_per_side ** 2         # 196

        if use_lora:
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["query", "key", "value"],
                lora_dropout=0.0,
                bias="none",
            )
            self._encoder = get_peft_model(self._encoder, lora_cfg)
            self._encoder.print_trainable_parameters()
        elif not finetune:
            for p in self._encoder.parameters():
                p.requires_grad_(False)
            self._encoder.eval()
        elif freeze_encoder_blocks > 0:
            # Freeze embeddings + first N blocks; only the tail adapts.
            n_blocks = len(self._encoder.encoder.layer)
            n_freeze = min(freeze_encoder_blocks, n_blocks)
            print(f"Freezing VideoMAE embeddings + first {n_freeze}/{n_blocks} encoder blocks.")
            for p in self._encoder.embeddings.parameters():
                p.requires_grad_(False)
            for block in self._encoder.encoder.layer[:n_freeze]:
                for p in block.parameters():
                    p.requires_grad_(False)

        self._encoder = torch.compile(self._encoder, mode="default")

    def train(self, mode=True):
        super().train(mode)
        if not self._finetune:
            self._encoder.eval()
        return self

    def forward(self, x):
        """
        x : [B, 3, T_in, H, W]  float32, already ImageNet-normalized
        Returns [B, T_model//2, hidden_dim]
        """
        B, C, T_in, H, W = x.shape
        assert T_in == self._T, (
            f"VideoMAEEncoder requires exactly {self._T} frames (obs_history={self._T}), got {T_in}"
        )

        # VideoMAE expects (B, T, C, H, W).
        x = x.permute(0, 2, 1, 3, 4)

        if self._finetune:
            tokens = self._encoder(pixel_values=x).last_hidden_state
        else:
            with torch.no_grad():
                tokens = self._encoder(pixel_values=x).last_hidden_state

        tokens = F.layer_norm(tokens, (self._hidden_dim,))

        T_enc = self._T // 2  # tubelet_size=2
        # Keep all spatial tokens: [B, T//2 * spatial_patches, hidden_dim]
        tokens = tokens.reshape(B, T_enc * self._spatial_patches, self._hidden_dim)
        return tokens

    @property
    def embed_dim(self):
        return self._hidden_dim

    @property
    def n_tokens(self):
        return (self._T // 2) * self._spatial_patches


class VideoMAEDiffusionTransformerAgent(DiffusionTransformerAgent):
    """
    DiffusionTransformerAgent with VideoMAE temporal visual encoding.

    Overrides embed() so the full [B, T, C, H, W] obs_history is passed as a
    single video clip [B, C, T, H, W] to the VideoMAE encoder.
    """

    def embed(self, imgs):
        """
        imgs : dict {"cam0": [B, T, 3, H, W], ...}
                     OR with "videomae_feats": [B, n_cams*(T//2), hidden_dim]
        Returns [B, n_cams * (T_model//2), hidden_dim]
        """
        if "videomae_feats" in imgs:
            return imgs["videomae_feats"]

        cam_features = (
            [self.visual_features] * self._n_cams
            if self._share_cam_features
            else list(self.visual_features)
        )
        clips = []
        for i, net in enumerate(cam_features):
            im = imgs[f"cam{i}"]             # [B, T, 3, H, W]
            im = im.permute(0, 2, 1, 3, 4)   # [B, 3, T, H, W]
            clips.append(net(im))            # [B, T_model//2, hidden_dim]
        return torch.cat(clips, dim=1)

    def make_pad_mask(self, obs_valid_len, B, device):
        return None


class BCPolicy_VideoMAE(nn.Module):
    """
    Diffusion-Transformer BC policy with a VideoMAE video encoder.

    The obs_history frames are encoded as a video clip by VideoMAE (HuggingFace),
    producing temporal patch tokens that condition the DiT diffusion denoiser.
    By default only the denoiser is trained; set finetune_videomae=True to also
    update the encoder at a lower learning rate.

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
        config_seq,
        config_rl,
        image_encoder_fn=lambda: None,
        **kwargs,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        cfg = config_rl.config_diffusion
        obs_history = cfg.get("obs_history", 16)

        finetune = cfg.get("finetune_videomae", False)
        use_lora = cfg.get("use_lora", False)
        encoder = VideoMAEEncoder(
            model_name=cfg.get("videomae_model", VIDEOMAE_MODEL),
            finetune=finetune,
            freeze_encoder_blocks=cfg.get("freeze_encoder_blocks", 0),
            use_lora=use_lora,
            lora_r=cfg.get("lora_r", 8),
            lora_alpha=cfg.get("lora_alpha", 16),
        )

        noise_net_kwargs = {
            **cfg.noise_net_kwargs.to_dict(),
            "cond_dim": encoder.embed_dim,
            "visual_cond_len": encoder.n_tokens * cfg.n_cams,
        }
        self.agent = VideoMAEDiffusionTransformerAgent(
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
            noise_net_class=DiTXNoiseNetAdapter,
        )
        self.agent.noise_net = torch.compile(self.agent.noise_net, mode="default")

        if finetune or use_lora:
            encoder_ids = {id(p) for p in encoder.parameters()}
            enc_params = [p for p in self.agent.parameters() if id(p) in encoder_ids and p.requires_grad]
            denoiser_params = [p for p in self.agent.parameters() if id(p) not in encoder_ids and p.requires_grad]
            enc_lr = cfg.get("videomae_lr", config_rl.actor_lr * 0.1)
            self.optimizer = Adam([
                {"params": denoiser_params, "lr": config_rl.actor_lr},
                {"params": enc_params, "lr": enc_lr},
            ])
        else:
            trainable = [p for p in self.agent.parameters() if p.requires_grad]
            self.optimizer = Adam(trainable, lr=config_rl.actor_lr)

        self.ema = EMAModel(
            model=self.agent,
            power=float(config_rl.get("ema_power", 0.75)),
            inv_gamma=float(config_rl.get("ema_inv_gamma", 1.0)),
            min_value=float(config_rl.get("ema_min_value", 0.0)),
            max_value=float(config_rl.get("ema_max_value", 0.9999)),
            update_after_step=int(config_rl.get("ema_update_after_step", 0)),
        )
        self.ema_shadow = self.ema.averaged_model
        self._live_agent_ref = [self.agent]

        # Feature cache — populated by precompute_frozen_cache(), only for frozen encoder.
        self._finetune = finetune
        self._use_cache = cfg.get("use_videomae_cache", True)
        self._encoder_ref = encoder
        self._n_cams = cfg.n_cams
        self._frozen_cache: torch.Tensor | None = None  # [max_buf_idx, n_cams*T//2, D], float16

    def precompute_frozen_cache(self, buffer, device, batch_size=64):
        """
        One-time pass over the full dataset to precompute and cache encoder outputs.
        Only valid when finetune=False. Call this once before the training loop.
        """
        if self._finetune or not self._use_cache:
            return
        if not (hasattr(buffer, "all_valid_starts") and hasattr(buffer, "get_windows_at_starts")):
            print("Buffer does not support feature caching — skipping precompute.")
            return

        from tqdm import trange
        all_starts = buffer.all_valid_starts()
        n_windows  = len(all_starts)
        max_idx    = int(all_starts.max()) + 1

        # float16 on CPU: ~12 KB/window for small model → ~480 MB for 40k windows
        cache = torch.zeros(
            max_idx,
            self._n_cams * self._encoder_ref.n_tokens,
            self._encoder_ref.embed_dim,
            dtype=torch.float16,
        )

        self._encoder_ref.eval()
        with torch.no_grad():
            for i in trange(0, n_windows, batch_size, desc="Caching VideoMAE features"):
                starts_i = all_starts[i : i + batch_size]
                batch    = buffer.get_windows_at_starts(starts_i)
                clips = []
                for c in range(self._n_cams):
                    im = batch["imgs"][f"cam{c}"].to(device)   # [B, T, C, H, W]
                    im = im.permute(0, 2, 1, 3, 4)             # [B, C, T, H, W]
                    clips.append(self._encoder_ref(im).cpu())  # [B, T//2, D]
                feats = torch.cat(clips, dim=1)                # [B, n_cams*T//2, D]
                cache[starts_i] = feats.half()

        self._frozen_cache = cache
        mb = cache.element_size() * cache.numel() / 1e6
        print(f"VideoMAE cache ready: {n_windows} windows, {mb:.0f} MB on CPU.")

    def forward(self, imgs, obs, actions, mask):
        B, ac_chunk, ac_dim = actions.shape
        ac_flat = actions.reshape(B, ac_chunk * ac_dim)
        mask_flat = mask.reshape(B, ac_chunk * ac_dim)
        return self.agent(imgs, obs, ac_flat, mask_flat)

    def update(self, batch):
        self.optimizer.zero_grad(set_to_none=True)

        imgs = batch["imgs"]
        if self._frozen_cache is not None and "buf_starts" in batch:
            starts = batch["buf_starts"]  # numpy int64 array, kept on CPU
            device = next(self.agent.parameters()).device
            imgs = dict(imgs)
            imgs["videomae_feats"] = (
                self._frozen_cache[starts].to(device).float()
            )

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = self.forward(imgs, batch["obs"], batch["act"], batch["mask"])
        loss.backward()
        self.optimizer.step()
        self.ema.step(self._live_agent_ref[0])
        return {"diffusion_loss": loss.item()}

    def eval(self):
        super().eval()
        self.agent = self.ema.averaged_model
        return self

    def train(self, mode=True):
        if mode:
            self.agent = self._live_agent_ref[0]
        super().train(mode)
        return self

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
