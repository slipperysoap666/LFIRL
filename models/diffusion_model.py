import math
from typing import Dict, Optional, Union, Tuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce
from diffusers import DDPMScheduler

Tensor = torch.Tensor
ObsType = Union[Tensor, Dict[str, Tensor]]


# Small helpers: dict_apply, ModuleAttrMixin, DictOfTensorMixin

def dict_apply(obj, fn):
    """
    Recursively apply fn to all non-dict leaves of a (possibly nested) dict.
    """
    if isinstance(obj, dict):
        return {k: dict_apply(v, fn) for k, v in obj.items()}
    return fn(obj)


class ModuleAttrMixin(nn.Module):
    """
    Minimal mixin to provide .device / .dtype for modules that don't
    explicitly register parameters (used by mask generators, etc.).
    """

    def __init__(self):
        # All subclasses must super().__init__()
        super().__init__()
        # Register a dummy buffer so .to(device) also affects this module
        self.register_buffer("_device_dummy", torch.empty(0), persistent=False)

    @property
    def device(self) -> torch.device:
        # Prefer parameters
        for p in self.parameters():
            return p.device
        # Then buffers (e.g. _device_dummy)
        for b in self.buffers():
            return b.device
        # Fallback
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        for p in self.parameters():
            return p.dtype
        for b in self.buffers():
            return b.dtype
        return torch.float32


class DictOfTensorMixin(nn.Module):
    """
    Base class for objects that keep a ParameterDict called params_dict.
    This is enough to support the LinearNormalizer API from the original code.
    """
    def __init__(self, params_dict: Optional[nn.ParameterDict] = None):
        super().__init__()
        if params_dict is None:
            params_dict = nn.ParameterDict()
        self.params_dict = params_dict


# LinearNormalizer (mostly identical to original repo)

class LinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ["limits", "gaussian"]

    @torch.no_grad()
    def fit(
        self,
        data: Union[Dict, torch.Tensor, "np.ndarray", "zarr.Array"],
        last_n_dims: int = 1,
        dtype=torch.float32,
        mode: str = "limits",
        output_max: float = 1.0,
        output_min: float = -1.0,
        range_eps: float = 1e-4,
        fit_offset: bool = True,
    ):
        if isinstance(data, dict):
            for key, value in data.items():
                self.params_dict[key] = _fit(
                    value,
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset,
                )
        else:
            self.params_dict["_default"] = _fit(
                data,
                last_n_dims=last_n_dims,
                dtype=dtype,
                mode=mode,
                output_max=output_max,
                output_min=output_min,
                range_eps=range_eps,
                fit_offset=fit_offset,
            )

    def __call__(self, x: Union[Dict, torch.Tensor, "np.ndarray"]) -> torch.Tensor:
        return self.normalize(x)

    def __getitem__(self, key: str):
        # __getitem__ is only called when we have checked the key exists
        return SingleFieldLinearNormalizer(self.params_dict[key])

    def __setitem__(self, key: str, value: "SingleFieldLinearNormalizer"):
        self.params_dict[key] = value.params_dict

    def _normalize_impl(self, x, forward: bool = True):
        if isinstance(x, dict):
            result = {}
            for key, value in x.items():
                params = self.params_dict[key]
                result[key] = _normalize(value, params, forward=forward)
            return result
        else:
            if "_default" not in self.params_dict:
                raise RuntimeError("LinearNormalizer not initialized")
            params = self.params_dict["_default"]
            return _normalize(x, params, forward=forward)

    def normalize(self, x: Union[Dict, torch.Tensor, "np.ndarray"]) -> torch.Tensor:
        return self._normalize_impl(x, forward=True)

    def unnormalize(self, x: Union[Dict, torch.Tensor, "np.ndarray"]) -> torch.Tensor:
        return self._normalize_impl(x, forward=False)

    def get_input_stats(self) -> Dict:
        if len(self.params_dict) == 0:
            raise RuntimeError("LinearNormalizer not initialized")
        if len(self.params_dict) == 1 and "_default" in self.params_dict:
            return self.params_dict["_default"]["input_stats"]

        result = {}
        for key, value in self.params_dict.items():
            if key != "_default":
                result[key] = value["input_stats"]
        return result

    def get_output_stats(self, key: str = "_default"):
        input_stats = self.get_input_stats()
        # no nested dict
        if "min" in input_stats:
            return dict_apply(input_stats, self.normalize)

        result = {}
        for k, group in input_stats.items():
            this_dict = {}
            for name, value in group.items():
                this_dict[name] = self.normalize({k: value})[k]
            result[k] = this_dict
        return result


class SingleFieldLinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ["limits", "gaussian"]

    @torch.no_grad()
    def fit(
        self,
        data: Union[torch.Tensor, "np.ndarray", "zarr.Array"],
        last_n_dims: int = 1,
        dtype=torch.float32,
        mode: str = "limits",
        output_max: float = 1.0,
        output_min: float = -1.0,
        range_eps: float = 1e-4,
        fit_offset: bool = True,
    ):
        self.params_dict = _fit(
            data,
            last_n_dims=last_n_dims,
            dtype=dtype,
            mode=mode,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            fit_offset=fit_offset,
        )

    @classmethod
    def create_fit(cls, data: Union[torch.Tensor, "np.ndarray", "zarr.Array"], **kwargs):
        obj = cls()
        obj.fit(data, **kwargs)
        return obj

    @classmethod
    def create_manual(
        cls,
        scale: Union[torch.Tensor, "np.ndarray"],
        offset: Union[torch.Tensor, "np.ndarray"],
        input_stats_dict: Dict[str, Union[torch.Tensor, "np.ndarray"]],
    ):
        def to_tensor(x):
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            x = x.flatten()
            return x

        # sanity check
        for x in [offset] + list(input_stats_dict.values()):
            assert x.shape == scale.shape
            assert x.dtype == scale.dtype

        params_dict = nn.ParameterDict(
            {
                "scale": to_tensor(scale),
                "offset": to_tensor(offset),
                "input_stats": nn.ParameterDict(dict_apply(input_stats_dict, to_tensor)),
            }
        )
        return cls(params_dict)

    @classmethod
    def create_identity(cls, dtype=torch.float32):
        scale = torch.tensor([1], dtype=dtype)
        offset = torch.tensor([0], dtype=dtype)
        input_stats_dict = {
            "min": torch.tensor([-1], dtype=dtype),
            "max": torch.tensor([1], dtype=dtype),
            "mean": torch.tensor([0], dtype=dtype),
            "std": torch.tensor([1], dtype=dtype),
        }
        return cls.create_manual(scale, offset, input_stats_dict)

    def normalize(self, x: Union[torch.Tensor, "np.ndarray"]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=True)

    def unnormalize(self, x: Union[torch.Tensor, "np.ndarray"]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=False)

    def get_input_stats(self):
        return self.params_dict["input_stats"]

    def get_output_stats(self):
        return dict_apply(self.params_dict["input_stats"], self.normalize)

    def __call__(self, x: Union[torch.Tensor, "np.ndarray"]) -> torch.Tensor:
        return self.normalize(x)


def _fit(
    data: Union[torch.Tensor, "np.ndarray", "zarr.Array"],
    last_n_dims: int = 1,
    dtype=torch.float32,
    mode: str = "limits",
    output_max: float = 1.0,
    output_min: float = -1.0,
    range_eps: float = 1e-4,
    fit_offset: bool = True,
):
    import numpy as np  # local import to avoid hard dependency if unused

    assert mode in ["limits", "gaussian"]
    assert last_n_dims >= 0
    assert output_max > output_min

    # convert data to torch and type
    try:
        import zarr
    except ImportError:
        zarr = None

    if zarr is not None and isinstance(data, zarr.Array):
        data = data[:]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if dtype is not None:
        data = data.type(dtype)

    # convert shape
    dim = 1
    if last_n_dims > 0:
        dim = int(np.prod(data.shape[-last_n_dims:]))
    data = data.reshape(-1, dim)

    # compute input stats min max mean std
    input_min, _ = data.min(axis=0)
    input_max, _ = data.max(axis=0)
    input_mean = data.mean(axis=0)
    input_std = data.std(axis=0)

    # compute scale and offset
    if mode == "limits":
        if fit_offset:
            # unit scale
            input_range = input_max - input_min
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
        else:
            # use this when data is pre-zero-centered.
            assert output_max > 0
            assert output_min < 0
            # unit abs
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(torch.abs(input_min), torch.abs(input_max))
            ignore_dim = input_abs < range_eps
            input_abs[ignore_dim] = output_abs
            # don't scale constant channels
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)
    elif mode == "gaussian":
        ignore_dim = input_std < range_eps
        scale = input_std.clone()
        scale[ignore_dim] = 1
        scale = 1 / scale

        if fit_offset:
            offset = -input_mean * scale
        else:
            offset = torch.zeros_like(input_mean)

    this_params = nn.ParameterDict(
        {
            "scale": scale,
            "offset": offset,
            "input_stats": nn.ParameterDict(
                {"min": input_min, "max": input_max, "mean": input_mean, "std": input_std}
            ),
        }
    )
    for p in this_params.parameters():
        p.requires_grad_(False)
    return this_params


def _normalize(x, params, forward: bool = True):
    import numpy as np  # local import

    assert "scale" in params
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    scale = params["scale"]
    offset = params["offset"]
    x = x.to(device=scale.device, dtype=scale.dtype)
    src_shape = x.shape
    x = x.reshape(-1, scale.shape[0])
    if forward:
        x = x * scale + offset
    else:
        x = (x - offset) / scale
    x = x.reshape(src_shape)
    return x


# Conv1d components & positional embedding (1D U-Net building blocks)

class Conv1dBlock(nn.Module):
    """
    Simple conv-norm-activation block used throughout the 1D UNet.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(num_groups=n_groups, num_channels=out_channels),
            nn.Mish(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Downsample1d(nn.Module):
    """
    1D downsampling via strided convolution (factor 2).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    """
    1D upsampling via transposed convolution (factor 2).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class SinusoidalPosEmb(nn.Module):
    """
    Standard sinusoidal positional embedding used for diffusion timesteps.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [B] int or float timesteps
        returns: [B, dim]
        """
        half = self.dim // 2
        device = x.device
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = x.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# Mask generators

class DummyMaskGenerator(ModuleAttrMixin):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, shape):
        device = self.device
        mask = torch.ones(size=shape, dtype=torch.bool, device=device)
        return mask


class LowdimMaskGenerator(ModuleAttrMixin):
    def __init__(
        self,
        action_dim,
        obs_dim,
        # obs mask setup
        max_n_obs_steps=2,
        fix_obs_steps=True,
        # action mask
        action_visible=False,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.max_n_obs_steps = max_n_obs_steps
        self.fix_obs_steps = fix_obs_steps
        self.action_visible = action_visible

    @torch.no_grad()
    def forward(self, shape, seed=None):
        device = self.device
        B, T, D = shape
        assert D == (self.action_dim + self.obs_dim)

        rng = torch.Generator(device=device)
        if seed is not None:
            rng = rng.manual_seed(seed)

        dim_mask = torch.zeros(size=shape, dtype=torch.bool, device=device)
        is_action_dim = dim_mask.clone()
        is_action_dim[..., : self.action_dim] = True
        is_obs_dim = ~is_action_dim

        if self.fix_obs_steps:
            obs_steps = torch.full((B,), fill_value=self.max_n_obs_steps, device=device)
        else:
            obs_steps = torch.randint(
                low=1,
                high=self.max_n_obs_steps + 1,
                size=(B,),
                generator=rng,
                device=device,
            )

        steps = torch.arange(0, T, device=device).reshape(1, T).expand(B, T)
        obs_mask = (steps.T < obs_steps).T.reshape(B, T, 1).expand(B, T, D)
        obs_mask = obs_mask & is_obs_dim

        if self.action_visible:
            action_steps = torch.maximum(
                obs_steps - 1,
                torch.tensor(0, dtype=obs_steps.dtype, device=obs_steps.device),
            )
            action_mask = (steps.T < action_steps).T.reshape(B, T, 1).expand(B, T, D)
            action_mask = action_mask & is_action_dim

        mask = obs_mask
        if self.action_visible:
            mask = mask | action_mask

        return mask


# Conditional UNet 1D

from einops.layers.torch import Rearrange  # after Conv1dBlock defined


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cond_dim,
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM modulation predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange("batch t -> batch t 1"),
        )

        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        x : [B, C_in, T]
        cond : [B, cond_dim]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    """
    1D UNet used in Stanford Diffusion Policy (low-dim) implementation.

    NOTE: input sample is (B, T, input_dim); internally we transpose to (B, input_dim, T)
    for Conv1d.
    """

    def __init__(
        self,
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
    ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList(
                [
                    # down encoder
                    ConditionalResidualBlock1D(
                        dim_in,
                        dim_out,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale,
                    ),
                    # up encoder
                    ConditionalResidualBlock1D(
                        dim_in,
                        dim_out,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale,
                    ),
                ]
            )

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
            ]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(
        self,
        sample: Tensor,
        timestep: Union[Tensor, float, int],
        local_cond: Optional[Tensor] = None,
        global_cond: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        sample: (B, T, input_dim)
        timestep: (B,) or scalar diffusion step
        local_cond: (B, T, local_cond_dim)
        global_cond: (B, global_cond_dim)
        output: (B, T, input_dim)
        """
        sample = rearrange(sample, "b t h -> b h t")

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            if global_cond.device != global_feature.device:
                global_cond = global_cond.to(global_feature.device)
            global_feature = torch.cat([global_feature, global_cond], dim=-1)

        h_local = []
        if local_cond is not None:
            if local_cond.device != sample.device:
                local_cond = local_cond.to(sample.device)
            local_cond = rearrange(local_cond, "b t h -> b h t")
            resnet, resnet2 = self.local_cond_encoder
            x_local = resnet(local_cond, global_feature)
            h_local.append(x_local)
            x_local = resnet2(local_cond, global_feature)
            h_local.append(x_local)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        x = rearrange(x, "b h t -> b t h")
        return x


# BaseLowdimPolicy

class BaseLowdimPolicy(ModuleAttrMixin):
    """
    Minimal base policy: forward ~ predict_action.
    """

    def __init__(self):
        super().__init__()

    def predict_action(self, obs_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        raise NotImplementedError

    def forward(self, obs_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.predict_action(obs_dict)


# DiffusionUnetLowdimPolicy (Stanford low-dim Diffusion Policy)

class DiffusionUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(
        self,
        model: ConditionalUnet1D,
        noise_scheduler: DDPMScheduler,
        horizon,
        obs_dim,
        action_dim,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_local_cond=False,
        obs_as_global_cond=False,
        pred_action_steps_only=False,
        oa_step_convention=False,
        **kwargs,
    ):
        super().__init__()

        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps


    def _has_field_stats(self, field: str) -> bool:
        """
        Check whether the normalizer has statistics for a specific field.
        """
        if not isinstance(self.normalizer, LinearNormalizer):
            return False
        pd = getattr(self.normalizer, "params_dict", None)
        if pd is None:
            return False
        return field in pd

    def _has_default_stats(self) -> bool:
        if not isinstance(self.normalizer, LinearNormalizer):
            return False
        pd = getattr(self.normalizer, "params_dict", None)
        if pd is None:
            return False
        return "_default" in pd

    def _normalize_obs(self, obs_tensor: Tensor) -> Tensor:
        """
        Normalize obs with the internal LinearNormalizer.

        For robustness:
        - If the normalizer has 'obs' stats, use them.
        - Else if it only has '_default', use that.
        - Else return obs as identity.
        """
        device = self.device
        if not isinstance(self.normalizer, LinearNormalizer):
            return obs_tensor.to(device)
        pd = getattr(self.normalizer, "params_dict", None)
        if pd is None or len(pd) == 0:
            return obs_tensor.to(device)

        try:
            if self._has_field_stats("obs"):
                nobs = self.normalizer["obs"].normalize(obs_tensor)
            elif self._has_default_stats():
                nobs = self.normalizer.normalize(obs_tensor)
            else:
                return obs_tensor.to(device)
            return nobs.to(device)
        except Exception:
            # Fallback: identity if anything goes wrong
            return obs_tensor.to(device)

    def _normalize_action(self, action_tensor: Tensor) -> Tensor:
        """
        Normalize actions (env space -> normalizer space).
        """
        device = self.device
        if not isinstance(self.normalizer, LinearNormalizer):
            return action_tensor.to(device)
        pd = getattr(self.normalizer, "params_dict", None)
        if pd is None or len(pd) == 0:
            return action_tensor.to(device)

        try:
            if self._has_field_stats("action"):
                na = self.normalizer["action"].normalize(action_tensor)
            elif self._has_default_stats():
                na = self.normalizer.normalize(action_tensor)
            else:
                return action_tensor.to(device)
            return na.to(device)
        except Exception:
            return action_tensor.to(device)

    def _unnormalize_action(self, naction_tensor: Tensor) -> Tensor:
        """
        Inverse-normalize actions (normalizer space -> env space).
        """
        device = self.device
        if not isinstance(self.normalizer, LinearNormalizer):
            return naction_tensor.to(device)
        pd = getattr(self.normalizer, "params_dict", None)
        if pd is None or len(pd) == 0:
            return naction_tensor.to(device)

        try:
            if self._has_field_stats("action"):
                act = self.normalizer["action"].unnormalize(naction_tensor)
            elif self._has_default_stats():
                act = self.normalizer.unnormalize(naction_tensor)
            else:
                return naction_tensor.to(device)
            return act.to(device)
        except Exception:
            return naction_tensor.to(device)

    def _get_action_scale_for_score(
        self,
        device: torch.device,
        dtype: torch.dtype,
        action_dim: int,
    ) -> Tensor:
        """
        Scale used to convert gradients from normalized action space to env space:

            d/da_env = d/da_norm * scale

        This helper REQUIRES that the normalizer has an "action" (or "_default")
        entry with a "scale" parameter whose dimension matches action_dim.
        If this is not the case, it raises a RuntimeError instead of silently
        falling back to identity, to avoid corrupting Q(s,a) gradient matching.
        """
        if not isinstance(self.normalizer, LinearNormalizer):
            raise RuntimeError(
                "Diffusion policy normalizer must be a LinearNormalizer before calling score_*."
            )

        pd = getattr(self.normalizer, "params_dict", None)
        if pd is None or len(pd) == 0:
            raise RuntimeError(
                "Diffusion policy normalizer has no parameters. "
                "Call model.normalizer.fit({'obs': ..., 'action': ...}, ...) "
                "before using score_*."
            )

        params = None
        if self._has_field_stats("action"):
            params = pd.get("action", None)
        elif self._has_default_stats():
            params = pd.get("_default", None)

        if params is None:
            raise RuntimeError(
                "Diffusion policy normalizer does not contain 'action' nor '_default' stats. "
                "Call model.normalizer.fit({'obs': ..., 'action': ...}, ...) before using score_*."
            )

        if "scale" not in params:
            raise RuntimeError("Normalizer params for 'action' do not contain 'scale'.")

        scale_vec = params["scale"].view(-1)

        if scale_vec.numel() != action_dim:
            raise RuntimeError(
                f"Normalizer 'action' scale dim {scale_vec.numel()} != action_dim {action_dim}."
            )

        scale = scale_vec.view(1, 1, -1).to(device=device, dtype=dtype)
        return scale


    @torch.no_grad()
    def conditional_sample(
        self,
        condition_data: Tensor,
        condition_mask: Tensor,
        local_cond: Optional[Tensor] = None,
        global_cond: Optional[Tensor] = None,
        generator=None,
        **kwargs,
    ) -> Tensor:
        """
        condition_data, condition_mask: (B, T, D)
        """
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(trajectory, t, local_cond=local_cond, global_cond=global_cond)
            trajectory = scheduler.step(model_output, t, trajectory, generator=generator, **kwargs).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert "obs" in obs_dict
        assert "past_action" not in obs_dict  # not implemented

        obs_tensor = obs_dict["obs"].to(self.device)
        nobs = self._normalize_obs(obs_tensor)

        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        device = self.device
        dtype = self.dtype

        local_cond = None
        global_cond = None

        if self.obs_as_local_cond:
            local_cond = torch.zeros(size=(B, T, Do), device=device, dtype=dtype)
            local_cond[:, :To] = nobs[:, :To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            global_cond = nobs[:, :To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            shape = (B, T, Da + Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs[:, :To]
            cond_mask[:, :To, Da:] = True

        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs,
        )

        naction_pred = nsample[..., :Da]
        action_pred = self._unnormalize_action(naction_pred)

        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

        result = {"action": action, "action_pred": action_pred}
        return result


    def set_normalizer(self, normalizer: LinearNormalizer):
        """
        Copy stats from an external normalizer and move them to the current device.
        """
        self.normalizer.load_state_dict(normalizer.state_dict())
        self.normalizer.to(self.device)

    def compute_loss(self, batch: Dict[str, Tensor]):
        """
        DDPM MSE epsilon loss over trajectory.

        - batch: {"obs": (B,T,Do), "action": (B,T_action,Da)}
        - if obs_as_global_cond=True, we condition on the first n_obs_steps
          of obs, and treat *full action sequence* as trajectory.
        """
        assert "valid_mask" not in batch

        nbatch = self.normalizer.normalize(batch)

        device = self.device
        for k, v in nbatch.items():
            if isinstance(v, torch.Tensor):
                nbatch[k] = v.to(device)

        obs = nbatch["obs"]
        action = nbatch["action"]

        local_cond = None
        global_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            local_cond = obs
            local_cond[:, self.n_obs_steps :, :] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:, : self.n_obs_steps, :].reshape(obs.shape[0], -1)
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        loss_mask = ~condition_mask

        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        pred = self.model(noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss


# High-level wrapper: dict / tensor obs + Q-gradient interface

class DiffusionPolicyTimeseries(DiffusionUnetLowdimPolicy):
    """
    High-level wrapper around DiffusionUnetLowdimPolicy:

    - Builds ConditionalUnet1D + DDPMScheduler.
    - Provides flatten_obs to support tensor or dict obs.
    - Provides sample_actions_from_obs for control from obs.
    - Provides score_first_action / score_actions:
      ∇_a log π(a | obs) in env action space, for Q(s,a) gradient matching.

    Assumes:
        obs_as_global_cond = True
        pred_action_steps_only = True
        obs_as_local_cond = False
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        num_train_timesteps: int = 1000,
        num_inference_steps: Optional[int] = None,
        obs_as_local_cond: bool = False,
        obs_as_global_cond: bool = True,
        pred_action_steps_only: bool = True,
        oa_step_convention: bool = False,
        diffusion_step_embed_dim: int = 256,
        down_dims: Sequence[int] = (256, 512, 1024),
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
        **kwargs,
    ):
        self._obs_key_order: Optional[Tuple[str, ...]] = None
        self.policy_type = "ddpm"

        if obs_as_local_cond or obs_as_global_cond:
            unet_input_dim = action_dim
        else:
            unet_input_dim = action_dim + obs_dim

        local_cond_dim = obs_dim if obs_as_local_cond else None
        global_cond_dim = obs_dim * n_obs_steps if obs_as_global_cond else None

        model = ConditionalUnet1D(
            input_dim=unet_input_dim,
            local_cond_dim=local_cond_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=1e-4,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            variance_type="fixed_small",
            clip_sample=True,
        )

        super().__init__(
            model=model,
            noise_scheduler=noise_scheduler,
            horizon=horizon,
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            num_inference_steps=num_inference_steps,
            obs_as_local_cond=obs_as_local_cond,
            obs_as_global_cond=obs_as_global_cond,
            pred_action_steps_only=pred_action_steps_only,
            oa_step_convention=oa_step_convention,
            **kwargs,
        )

    # obs flatten: support dict / tensor, fixed key order
    def flatten_obs(self, obs: ObsType) -> Tensor:
        """
        Flatten obs to Tensor[B, T, Do], where Do == self.obs_dim.

        - If obs is Tensor[B,T,S], returns it after checking S == obs_dim.
        - If obs is Dict[str, Tensor]:
            * On first call, records sorted(keys) as _obs_key_order.
            * Later calls use the same key order, ignoring extra keys
              and requiring all recorded keys to be present.
        """
        if isinstance(obs, torch.Tensor):
            if obs.dim() != 3:
                raise ValueError(f"Tensor obs must be [B, T, S], got {tuple(obs.shape)}")
            if obs.shape[-1] != self.obs_dim:
                raise ValueError(
                    f"Tensor obs last dim {obs.shape[-1]} != expected obs_dim {self.obs_dim}"
                )
            return obs.to(self.device)

        if not isinstance(obs, dict):
            raise TypeError("obs must be Tensor or Dict[str, Tensor]")

        if self._obs_key_order is None:
            keys = sorted(obs.keys())
            self._obs_key_order = tuple(keys)
        else:
            keys = self._obs_key_order

        parts = []
        B = None
        T = None

        for k in keys:
            if k not in obs:
                raise KeyError(
                    f"flatten_obs expected obs key '{k}' (from training), "
                    f"but it's missing in current obs dict. "
                    f"Available keys now: {list(obs.keys())}"
                )
            v = obs[k]
            if not isinstance(v, torch.Tensor):
                v = torch.as_tensor(v, dtype=torch.float32, device=self.device)
            else:
                v = v.to(self.device)

            if v.dim() == 2:
                v = v.unsqueeze(1)  # [B,Sk] -> [B,1,Sk]
            elif v.dim() != 3:
                raise ValueError(f"obs['{k}'] must be [B,T,Sk] or [B,Sk], got {tuple(v.shape)}")

            if B is None:
                B, T = v.shape[0], v.shape[1]
            else:
                if v.shape[0] != B or v.shape[1] != T:
                    raise ValueError(
                        f"All obs components must share batch/time dims; "
                        f"key '{k}' has shape {tuple(v.shape)}, expected B={B}, T={T}."
                    )
            parts.append(v)

        flat = torch.cat(parts, dim=-1)
        if flat.shape[-1] != self.obs_dim:
            raise ValueError(
                f"flatten_obs produced obs_dim={flat.shape[-1]}, expected {self.obs_dim}. "
                f"Used keys={keys}."
            )
        return flat

    # Inference: sample actions from obs
    @torch.no_grad()
    def sample_actions_from_obs(self, obs: ObsType) -> Tensor:
        """
        obs: tensor or dict, length >= n_obs_steps.
        return: actions [B, n_action_steps, Da]
        """
        self.eval()
        obs_flat = self.flatten_obs(obs)
        B, To, Do = obs_flat.shape
        if Do != self.obs_dim:
            raise ValueError(f"obs_dim mismatch: got {Do}, expected {self.obs_dim}")

        if To < self.n_obs_steps:
            pad = obs_flat[:, -1:, :].expand(B, self.n_obs_steps - To, Do)
            obs_cond = torch.cat([obs_flat, pad], dim=1)
        else:
            obs_cond = obs_flat[:, : self.n_obs_steps, :]

        if self.horizon < self.n_obs_steps:
            raise ValueError("horizon must be >= n_obs_steps")
        obs_pad = torch.zeros(
            B,
            self.horizon,
            Do,
            device=obs_flat.device,
            dtype=obs_flat.dtype,
        )
        obs_pad[:, : self.n_obs_steps, :] = obs_cond

        obs_dict = {"obs": obs_pad}
        out = self.predict_action(obs_dict)
        return out["action"]

    @torch.no_grad()
    def sample_actions(self, obs: ObsType, device: Optional[torch.device] = None) -> Tensor:
        """
        Backward-compatible helper:
            model.sample_actions(obs_seq, device=device) -> [B, Tp, A]
        """
        if device is not None:
            if isinstance(obs, dict):
                obs = {k: v.to(device) for k, v in obs.items()}
            elif isinstance(obs, torch.Tensor):
                obs = obs.to(device)
        return self.sample_actions_from_obs(obs)

    # Q-learning: score_first_action
    @torch.no_grad()
    def score_first_action(
        self,
        obs: ObsType,
        action_seq: Tensor,
        t_eval: Optional[int] = None,
        n_noise: int = 1,
    ) -> Tensor:
        """
        Compute ∇_a log π(a|obs) at the first action in the sequence (env space).

        obs:   Tensor[B,To,Do] or Dict[str,Tensor], To >= n_obs_steps
        action_seq: Tensor[B, n_action_steps, Da] in env action space
        t_eval: fixed diffusion timestep if not None; else sampled uniformly
        n_noise: # Monte-Carlo samples over noise
        """
        if not (self.obs_as_global_cond and self.pred_action_steps_only and not self.obs_as_local_cond):
            raise RuntimeError(
                "score_first_action assumes obs_as_global_cond=True, "
                "pred_action_steps_only=True, obs_as_local_cond=False."
            )

        self.eval()

        obs_flat = self.flatten_obs(obs)
        B, To, Do = obs_flat.shape
        if Do != self.obs_dim:
            raise ValueError(f"obs_dim mismatch: got {Do}, expected {self.obs_dim}")

        if To < self.n_obs_steps:
            pad = obs_flat[:, -1:, :].expand(B, self.n_obs_steps - To, Do)
            obs_cond = torch.cat([obs_flat, pad], dim=1)
        else:
            obs_cond = obs_flat[:, : self.n_obs_steps, :]

        device = obs_flat.device
        dtype = obs_flat.dtype

        if action_seq.dim() != 3:
            raise ValueError(f"action_seq must be [B, n_action_steps, Da], got {tuple(action_seq.shape)}")
        if action_seq.shape[0] != B:
            raise ValueError("action_seq batch size must match obs batch size")
        if action_seq.shape[1] != self.n_action_steps:
            raise ValueError(
                f"action_seq length {action_seq.shape[1]} != n_action_steps {self.n_action_steps}"
            )
        if action_seq.shape[2] != self.action_dim:
            raise ValueError(
                f"action_dim mismatch: got {action_seq.shape[2]}, expected {self.action_dim}"
            )

        nobs = self._normalize_obs(obs_cond)
        nactions = self._normalize_action(action_seq.to(device))

        global_cond = nobs[:, : self.n_obs_steps, :].reshape(B, -1)

        x0 = nactions
        B, Tp, Da = x0.shape

        num_train_steps = self.noise_scheduler.config.num_train_timesteps
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=device, dtype=dtype)

        grad_accum = torch.zeros(B, Da, device=device, dtype=dtype)

        for _ in range(n_noise):
            if t_eval is None:
                t = torch.randint(0, num_train_steps, (B,), device=device, dtype=torch.long)
            else:
                t = torch.full((B,), int(t_eval), device=device, dtype=torch.long)

            noise = torch.randn_like(x0)
            x_t = self.noise_scheduler.add_noise(x0, noise, t)
            eps_pred = self.model(x_t, t, local_cond=None, global_cond=global_cond)

            alpha_bar = alphas_cumprod[t]
            sqrt_alpha_bar = alpha_bar.sqrt().view(B, 1, 1)
            sigma_t = (1.0 - alpha_bar).sqrt().view(B, 1, 1)

            score_xt = -eps_pred / (sigma_t + 1e-8)
            score_x0 = sqrt_alpha_bar * score_xt

            scale = self._get_action_scale_for_score(
                device=score_x0.device,
                dtype=score_x0.dtype,
                action_dim=self.action_dim,
            )
            score_env = score_x0 * scale

            grad_accum += score_env[:, 0, :]

        grad_first = grad_accum / float(n_noise)
        return grad_first

    # Full sequence gradient (optional)
    @torch.no_grad()
    def score_actions(
        self,
        obs: ObsType,
        action_seq: Tensor,
        t_eval: Optional[int] = None,
        n_noise: int = 1,
    ) -> Tensor:
        """
        Same as score_first_action, but returns gradients for all time steps:

            grad_seq: [B, n_action_steps, Da]
        """
        if not (self.obs_as_global_cond and self.pred_action_steps_only and not self.obs_as_local_cond):
            raise RuntimeError(
                "score_actions assumes obs_as_global_cond=True, "
                "pred_action_steps_only=True, obs_as_local_cond=False."
            )

        self.eval()

        obs_flat = self.flatten_obs(obs)
        B, To, Do = obs_flat.shape
        if Do != self.obs_dim:
            raise ValueError(f"obs_dim mismatch: got {Do}, expected {self.obs_dim}")

        if To < self.n_obs_steps:
            pad = obs_flat[:, -1:, :].expand(B, self.n_obs_steps - To, Do)
            obs_cond = torch.cat([obs_flat, pad], dim=1)
        else:
            obs_cond = obs_flat[:, : self.n_obs_steps, :]

        device = obs_flat.device
        dtype = obs_flat.dtype

        if action_seq.dim() != 3:
            raise ValueError(f"action_seq must be [B, n_action_steps, Da], got {tuple(action_seq.shape)}")
        if action_seq.shape[0] != B:
            raise ValueError("action_seq batch size must match obs batch size")
        if action_seq.shape[1] != self.n_action_steps:
            raise ValueError(
                f"action_seq length {action_seq.shape[1]} != n_action_steps {self.n_action_steps}"
            )
        if action_seq.shape[2] != self.action_dim:
            raise ValueError(
                f"action_dim mismatch: got {action_seq.shape[2]}, expected {self.action_dim}"
            )

        nobs = self._normalize_obs(obs_cond)
        nactions = self._normalize_action(action_seq.to(device))

        global_cond = nobs[:, : self.n_obs_steps, :].reshape(B, -1)
        x0 = nactions
        B, Tp, Da = x0.shape

        num_train_steps = self.noise_scheduler.config.num_train_timesteps
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=device, dtype=dtype)

        grad_accum = torch.zeros(B, Tp, Da, device=device, dtype=dtype)

        for _ in range(n_noise):
            if t_eval is None:
                t = torch.randint(0, num_train_steps, (B,), device=device, dtype=torch.long)
            else:
                t = torch.full((B,), int(t_eval), device=device, dtype=torch.long)

            noise = torch.randn_like(x0)
            x_t = self.noise_scheduler.add_noise(x0, noise, t)
            eps_pred = self.model(x_t, t, local_cond=None, global_cond=global_cond)

            alpha_bar = alphas_cumprod[t]
            sqrt_alpha_bar = alpha_bar.sqrt().view(B, 1, 1)
            sigma_t = (1.0 - alpha_bar).sqrt().view(B, 1, 1)

            score_xt = -eps_pred / (sigma_t + 1e-8)
            score_x0 = sqrt_alpha_bar * score_xt

            scale = self._get_action_scale_for_score(
                device=score_x0.device,
                dtype=score_x0.dtype,
                action_dim=self.action_dim,
            )
            score_env = score_x0 * scale

            grad_accum += score_env

        grad_seq = grad_accum / float(n_noise)
        return grad_seq


class FlowMatchingPolicyTimeseries(DiffusionPolicyTimeseries):
    """
    Flow Matching variant that keeps the Stanford low-dim DP backbone
    (ConditionalUnet1D + global observation conditioning) unchanged.

    Differences from DDPM:
    - Training uses straight-line conditional flow matching.
    - Sampling integrates the learned velocity field with Euler steps.
    - score_* returns a score proxy derived from the learned flow field.
    """

    def __init__(self, *args, flow_epsilon: float = 1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_type = "flow_matching"
        self.flow_epsilon = float(flow_epsilon)

    def _sample_flow_times(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        t = torch.rand(batch_size, device=device, dtype=dtype)
        eps = max(0.0, min(0.499, self.flow_epsilon))
        if eps > 0.0:
            t = t * (1.0 - 2.0 * eps) + eps
        return t

    def _flow_to_model_time(self, t: Tensor) -> Tensor:
        """
        Map normalized flow time in [0, 1] to the timestep scale expected by the
        original diffusion timestep embedding.
        """
        num_train_steps = max(1, int(self.noise_scheduler.config.num_train_timesteps) - 1)
        return t * float(num_train_steps)

    @torch.no_grad()
    def conditional_sample(
        self,
        condition_data: Tensor,
        condition_mask: Tensor,
        local_cond: Optional[Tensor] = None,
        global_cond: Optional[Tensor] = None,
        generator=None,
        **kwargs,
    ) -> Tensor:
        del generator, kwargs

        model = self.model
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
        )

        num_steps = max(1, int(self.num_inference_steps))
        times = torch.linspace(
            0.0,
            1.0,
            num_steps + 1,
            device=condition_data.device,
            dtype=condition_data.dtype,
        )

        for step_idx in range(num_steps):
            trajectory[condition_mask] = condition_data[condition_mask]

            t_cur = trajectory.new_full((trajectory.shape[0],), float(times[step_idx].item()))
            model_t = self._flow_to_model_time(t_cur)
            velocity = model(trajectory, model_t, local_cond=local_cond, global_cond=global_cond)

            dt = times[step_idx + 1] - times[step_idx]
            trajectory = trajectory + dt * velocity

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def compute_loss(self, batch: Dict[str, Tensor]):
        """
        Straight-line conditional flow matching loss over trajectory.
        """
        assert "valid_mask" not in batch

        nbatch = self.normalizer.normalize(batch)

        device = self.device
        for k, v in nbatch.items():
            if isinstance(v, torch.Tensor):
                nbatch[k] = v.to(device)

        obs = nbatch["obs"]
        action = nbatch["action"]

        local_cond = None
        global_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            local_cond = obs
            local_cond[:, self.n_obs_steps :, :] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:, : self.n_obs_steps, :].reshape(obs.shape[0], -1)
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        x0 = trajectory
        x1 = torch.randn_like(x0)
        bsz = x0.shape[0]

        t = self._sample_flow_times(bsz, device=x0.device, dtype=x0.dtype)
        t_view = t.view(bsz, 1, 1)

        x_t = (1.0 - t_view) * x1 + t_view * x0
        x_t[condition_mask] = x0[condition_mask]

        target_velocity = x0 - x1
        pred_velocity = self.model(
            x_t,
            self._flow_to_model_time(t),
            local_cond=local_cond,
            global_cond=global_cond,
        )

        loss_mask = ~condition_mask
        loss = F.mse_loss(pred_velocity, target_velocity, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss

    @torch.no_grad()
    def score_first_action(
        self,
        obs: ObsType,
        action_seq: Tensor,
        t_eval: Optional[int] = None,
        n_noise: int = 1,
    ) -> Tensor:
        if not (self.obs_as_global_cond and self.pred_action_steps_only and not self.obs_as_local_cond):
            raise RuntimeError(
                "score_first_action assumes obs_as_global_cond=True, "
                "pred_action_steps_only=True, obs_as_local_cond=False."
            )

        self.eval()

        obs_flat = self.flatten_obs(obs)
        B, To, Do = obs_flat.shape
        if Do != self.obs_dim:
            raise ValueError(f"obs_dim mismatch: got {Do}, expected {self.obs_dim}")

        if To < self.n_obs_steps:
            pad = obs_flat[:, -1:, :].expand(B, self.n_obs_steps - To, Do)
            obs_cond = torch.cat([obs_flat, pad], dim=1)
        else:
            obs_cond = obs_flat[:, : self.n_obs_steps, :]

        device = obs_flat.device
        dtype = obs_flat.dtype

        if action_seq.dim() != 3:
            raise ValueError(f"action_seq must be [B, n_action_steps, Da], got {tuple(action_seq.shape)}")
        if action_seq.shape[0] != B:
            raise ValueError("action_seq batch size must match obs batch size")
        if action_seq.shape[1] != self.n_action_steps:
            raise ValueError(
                f"action_seq length {action_seq.shape[1]} != n_action_steps {self.n_action_steps}"
            )
        if action_seq.shape[2] != self.action_dim:
            raise ValueError(
                f"action_dim mismatch: got {action_seq.shape[2]}, expected {self.action_dim}"
            )

        nobs = self._normalize_obs(obs_cond)
        nactions = self._normalize_action(action_seq.to(device))

        global_cond = nobs[:, : self.n_obs_steps, :].reshape(B, -1)

        x0 = nactions
        _, _, Da = x0.shape

        grad_accum = torch.zeros(B, Da, device=device, dtype=dtype)
        denom_eps = max(self.flow_epsilon, 1e-4)
        num_train_steps = max(1, int(self.noise_scheduler.config.num_train_timesteps))

        for _ in range(n_noise):
            if t_eval is None:
                t = self._sample_flow_times(B, device=device, dtype=dtype)
            else:
                t_scalar = (float(int(t_eval)) + 0.5) / float(num_train_steps)
                t = torch.full((B,), t_scalar, device=device, dtype=dtype)
                t = t.clamp(min=denom_eps, max=1.0 - denom_eps)

            t_view = t.view(B, 1, 1)
            noise = torch.randn_like(x0)
            x_t = (1.0 - t_view) * noise + t_view * x0

            velocity = self.model(
                x_t,
                self._flow_to_model_time(t),
                local_cond=None,
                global_cond=global_cond,
            )

            # For the straight path x_t = t * x0 + (1 - t) * x1, use the
            # induced Gaussian bridge to obtain a score proxy in x0 space.
            score_xt = (t_view * velocity - x_t) / (1.0 - t_view + denom_eps)
            score_x0 = t_view * score_xt

            scale = self._get_action_scale_for_score(
                device=score_x0.device,
                dtype=score_x0.dtype,
                action_dim=self.action_dim,
            )
            score_env = score_x0 * scale

            grad_accum += score_env[:, 0, :]

        grad_first = grad_accum / float(n_noise)
        return grad_first

    @torch.no_grad()
    def score_actions(
        self,
        obs: ObsType,
        action_seq: Tensor,
        t_eval: Optional[int] = None,
        n_noise: int = 1,
    ) -> Tensor:
        if not (self.obs_as_global_cond and self.pred_action_steps_only and not self.obs_as_local_cond):
            raise RuntimeError(
                "score_actions assumes obs_as_global_cond=True, "
                "pred_action_steps_only=True, obs_as_local_cond=False."
            )

        self.eval()

        obs_flat = self.flatten_obs(obs)
        B, To, Do = obs_flat.shape
        if Do != self.obs_dim:
            raise ValueError(f"obs_dim mismatch: got {Do}, expected {self.obs_dim}")

        if To < self.n_obs_steps:
            pad = obs_flat[:, -1:, :].expand(B, self.n_obs_steps - To, Do)
            obs_cond = torch.cat([obs_flat, pad], dim=1)
        else:
            obs_cond = obs_flat[:, : self.n_obs_steps, :]

        device = obs_flat.device
        dtype = obs_flat.dtype

        if action_seq.dim() != 3:
            raise ValueError(f"action_seq must be [B, n_action_steps, Da], got {tuple(action_seq.shape)}")
        if action_seq.shape[0] != B:
            raise ValueError("action_seq batch size must match obs batch size")
        if action_seq.shape[1] != self.n_action_steps:
            raise ValueError(
                f"action_seq length {action_seq.shape[1]} != n_action_steps {self.n_action_steps}"
            )
        if action_seq.shape[2] != self.action_dim:
            raise ValueError(
                f"action_dim mismatch: got {action_seq.shape[2]}, expected {self.action_dim}"
            )

        nobs = self._normalize_obs(obs_cond)
        nactions = self._normalize_action(action_seq.to(device))

        global_cond = nobs[:, : self.n_obs_steps, :].reshape(B, -1)
        x0 = nactions
        _, Tp, Da = x0.shape

        grad_accum = torch.zeros(B, Tp, Da, device=device, dtype=dtype)
        denom_eps = max(self.flow_epsilon, 1e-4)
        num_train_steps = max(1, int(self.noise_scheduler.config.num_train_timesteps))

        for _ in range(n_noise):
            if t_eval is None:
                t = self._sample_flow_times(B, device=device, dtype=dtype)
            else:
                t_scalar = (float(int(t_eval)) + 0.5) / float(num_train_steps)
                t = torch.full((B,), t_scalar, device=device, dtype=dtype)
                t = t.clamp(min=denom_eps, max=1.0 - denom_eps)

            t_view = t.view(B, 1, 1)
            noise = torch.randn_like(x0)
            x_t = (1.0 - t_view) * noise + t_view * x0

            velocity = self.model(
                x_t,
                self._flow_to_model_time(t),
                local_cond=None,
                global_cond=global_cond,
            )

            score_xt = (t_view * velocity - x_t) / (1.0 - t_view + denom_eps)
            score_x0 = t_view * score_xt

            scale = self._get_action_scale_for_score(
                device=score_x0.device,
                dtype=score_x0.dtype,
                action_dim=self.action_dim,
            )
            score_env = score_x0 * scale

            grad_accum += score_env

        grad_seq = grad_accum / float(n_noise)
        return grad_seq
