# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import copy

import torch
from torch import nn


class _BatchNorm1DHelper(nn.BatchNorm1d):
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.transpose(1, 2)
            x = super().forward(x)
            return x.transpose(1, 2)
        return super().forward(x)


class BaseAgent(nn.Module):
    def __init__(
        self,
        odim,
        features,
        n_cams,
        imgs_per_cam,
        use_obs=False,
        share_cam_features=False,
        early_fusion=False,
        dropout=0,
        feat_norm=None,
        token_dim=None,
        obs_history=1,
    ):
        super().__init__()

        # store visual features (duplicate weights if shared)
        self._n_cams = n_cams
        self._share_cam_features = share_cam_features
        if self._share_cam_features:
            self.visual_features = features
        else:
            feat_list = [features] + [copy.deepcopy(features) for _ in range(1, n_cams)]
            self.visual_features = nn.ModuleList(feat_list)

        self.early_fusion = early_fusion
        imgs_per_cam = 1 if early_fusion else imgs_per_cam
        self._token_dim = features.embed_dim
        self._n_tokens = imgs_per_cam * n_cams * features.n_tokens
        self._obs_history = obs_history
        self._n_img_toks_per_frame = 0 if early_fusion else n_cams * features.n_tokens

        # handle obs tokenization strategies
        if use_obs == "add_token":
            self._obs_strat = "add_token"
            self._n_tokens += obs_history  # one token per history frame
            self._obs_proc = nn.Sequential(
                nn.Dropout(p=0.2), nn.Linear(odim, self._token_dim)
            )
        elif use_obs == "pad_img_tokens":
            self._obs_strat = "pad_img_tokens"
            self._token_dim += odim
            self._obs_proc = nn.Dropout(p=0.2)
        else:
            assert not use_obs
            self._obs_strat = None

        # build (optional) token feature projection layer
        linear_proj = nn.Identity()
        if token_dim is not None and token_dim != self._token_dim:
            linear_proj = nn.Linear(self._token_dim, token_dim)
            self._token_dim = token_dim

        # build feature normalization layers
        if feat_norm == "batch_norm":
            norm = _BatchNorm1DHelper(self._token_dim)
        elif feat_norm == "layer_norm":
            norm = nn.LayerNorm(self._token_dim)
        else:
            assert feat_norm is None
            norm = nn.Identity()

        # final token post proc network
        self.post_proc = nn.Sequential(linear_proj, norm, nn.Dropout(dropout))

    def make_pad_mask(self, obs_valid_len, B, device):
        """
        Build a (B, n_tokens) bool mask for the encoder where True = padding
        (ignored in attention). Only the first (obs_history - obs_valid_len)
        frames are treated as padding.
        """
        T = self._obs_history
        n_pad = max(0, T - obs_valid_len)
        if n_pad == 0:
            return None

        # frame-level: True for the n_pad oldest (zero-padded) frames
        frame_pad = torch.zeros(T, dtype=torch.bool, device=device)
        frame_pad[:n_pad] = True

        # expand to image-token level
        img_mask = frame_pad.repeat_interleave(self._n_img_toks_per_frame)

        if self._obs_strat == "add_token":
            # obs tokens are appended after all image tokens, one per frame
            mask = torch.cat([img_mask, frame_pad])
        else:
            mask = img_mask

        return mask.unsqueeze(0).expand(B, -1)  # (B, n_tokens)

    def forward(self, imgs, obs, ac_flat, mask_flat):
        raise NotImplementedError

    def get_actions(self, img, obs):
        raise NotImplementedError

    def tokenize_obs(self, imgs, obs, flatten=False):
        # start by getting image tokens
        tokens = self.embed(imgs)

        if self._obs_strat == "add_token":
            # obs: (B, obs_dim) or (B, T, obs_dim)
            obs_tokens = self._obs_proc(obs)  # (B, token_dim) or (B, T, token_dim)
            if obs_tokens.dim() == 2:
                obs_tokens = obs_tokens[:, None]  # (B, 1, token_dim)
            tokens = torch.cat((tokens, obs_tokens), 1)
        elif self._obs_strat == "pad_img_tokens":
            # use only the last obs frame for padding
            if obs.dim() == 3:
                obs = obs[:, -1]  # (B, obs_dim)
            obs = self._obs_proc(obs)
            obs = obs[:, None].repeat((1, tokens.shape[1], 1))
            tokens = torch.cat((obs, tokens), 2)
        else:
            assert self._obs_strat is None

        tokens = self.post_proc(tokens)
        if flatten:
            return tokens.reshape((tokens.shape[0], -1))
        return tokens

    def embed(self, imgs):
        def embed_helper(net, im):
            if self.early_fusion and len(im.shape) == 5:
                T = im.shape[1]
                im = torch.cat([im[:, t] for t in range(T)], 1)
                return net(im)
            elif len(im.shape) == 5:
                B, T, C, H, W = im.shape
                if T == 0:
                    return torch.empty(B, 0, net.embed_dim, device=im.device, dtype=im.dtype)
                if T > 1:
                    # split on the time axis before flattening so history and
                    # current frames are correctly separated across all batch items
                    hist_frames = im[:, :-1].reshape(B * (T - 1), C, H, W)
                    curr_frames = im[:, -1]  # (B, C, H, W)
                    with torch.no_grad():
                        hist = net(hist_frames)              # (B*(T-1), n_tokens, embed_dim)
                    curr = net(curr_frames)                  # (B, n_tokens, embed_dim)
                    hist = hist.reshape(B, -1, net.embed_dim)  # (B, (T-1)*n_tokens, embed_dim)
                    embeds = torch.cat([hist, curr], dim=1)  # (B, T*n_tokens, embed_dim)
                else:
                    embeds = net(im.reshape(B, C, H, W))     # (B, n_tokens, embed_dim)
                return embeds.reshape(B, -1, net.embed_dim)

            assert len(im.shape) == 4
            return net(im)

        if self._share_cam_features:
            embeds = [
                embed_helper(self.visual_features, imgs[f"cam{i}"])
                for i in range(self._n_cams)
            ]
        else:
            embeds = [
                embed_helper(net, imgs[f"cam{i}"])
                for i, net in enumerate(self.visual_features)
            ]
        return torch.cat(embeds, dim=1)

    @property
    def n_cams(self):
        return self._n_cams

    @property
    def ac_chunk(self):
        raise NotImplementedError

    @property
    def token_dim(self):
        return self._token_dim

    @property
    def n_tokens(self):
        return self._n_tokens
