import torch
from torch.nn.modules.batchnorm import _BatchNorm


class EMAModel:
    """
    Exponential moving average of model weights, ported from
    diffusion_policy.model.diffusion.ema_model.EMAModel (the LDP / DP repo)
    so that decay scheduling matches their experiments exactly.
    """

    def __init__(
        self,
        model,
        update_after_step=0,
        inv_gamma=1.0,
        power=0.75,
        min_value=0.0,
        max_value=0.9999,
    ):
        import copy
        self.averaged_model = copy.deepcopy(model)
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        if step <= 0:
            return 0.0
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        # Ensure the shadow lives on the same device/dtype as the live model.
        # `copy.deepcopy` happens before BCPolicy_LDP is moved to CUDA, so the
        # shadow starts on CPU; sync it on the first step (cheap no-op after).
        live_param = next(new_model.parameters(), None)
        if live_param is not None:
            ema_param = next(self.averaged_model.parameters(), None)
            if ema_param is not None and ema_param.device != live_param.device:
                self.averaged_model.to(live_param.device)

        self.decay = self.get_decay(self.optimization_step)
        for module, ema_module in zip(
            new_model.modules(), self.averaged_model.modules()
        ):
            for param, ema_param in zip(
                module.parameters(recurse=False),
                ema_module.parameters(recurse=False),
            ):
                if isinstance(param, dict):
                    raise RuntimeError("Dict parameter not supported")
                if isinstance(module, _BatchNorm) or not param.requires_grad:
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    ema_param.mul_(self.decay)
                    ema_param.add_(
                        param.data.to(dtype=ema_param.dtype),
                        alpha=1 - self.decay,
                    )
        self.optimization_step += 1
