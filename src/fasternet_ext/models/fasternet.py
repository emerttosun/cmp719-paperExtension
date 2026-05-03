"""FasterNet-T0/T1 with optional Channel Gate Module.

This implementation follows the official FasterNet classification block at a
small, dependency-light scale suitable for CIFAR-100 experiments. The extension
is intentionally scoped: ChannelGateModule is applied only to the channel subset
processed by PConv.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Iterable, List, Literal, Optional

import torch
from torch import Tensor, nn


CGMPlacement = Literal["none", "all", "early", "late", "s1", "s2", "s3", "s4", "s2s3", "s2s4"]
CGMMode = Literal["sigmoid", "residual"]
CGMType = Literal["se", "eca"]
CGMPooling = Literal["gap", "gap_gmp"]


class DropPath(nn.Module):
    """Stochastic depth per sample.

    Kept local to avoid a timm dependency for the course-project baseline.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ChannelGateModule(nn.Module):
    """SE-style gate for only the PConv-processed channel subset."""

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        mode: CGMMode = "sigmoid",
        alpha: float = 0.5,
        gate_type: CGMType = "se",
        eca_kernel_size: int = 3,
        pooling: CGMPooling = "gap",
        init_bias: float = 0.0,
    ) -> None:
        super().__init__()
        if mode not in {"sigmoid", "residual"}:
            raise ValueError(f"Unsupported CGM mode: {mode}")
        if gate_type not in {"se", "eca"}:
            raise ValueError(f"Unsupported CGM gate type: {gate_type}")
        if pooling not in {"gap", "gap_gmp"}:
            raise ValueError(f"Unsupported CGM pooling: {pooling}")
        if eca_kernel_size % 2 == 0:
            raise ValueError("ECA kernel size must be odd.")
        self.mode = mode
        self.alpha = alpha
        self.gate_type = gate_type
        self.pooling = pooling
        self.init_bias = float(init_bias)
        if self.init_bias > 0.0 and mode != "sigmoid":
            raise ValueError("CGM identity init is only supported with cgm_mode='sigmoid'.")
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        if gate_type == "eca":
            # Keep bias disabled by default so prior ECA experiments remain reproducible.
            self.gate = nn.Conv1d(
                1,
                1,
                kernel_size=eca_kernel_size,
                padding=eca_kernel_size // 2,
                bias=self.init_bias > 0.0,
            )
        else:
            self.gate = nn.Sequential(
                nn.Conv2d(channels, hidden, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, channels, kernel_size=1),
            )
        self.latest_gate: Optional[Tensor] = None
        self.latest_scale: Optional[Tensor] = None

    @torch.no_grad()
    def identity_init(self) -> None:
        """Initialize the final gate layer so sigmoid output starts near 1.0.

        With init_bias = b > 0, the gate output layer is reset to weight=0 and
        bias=b. For GAP+GMP pooling the per-branch bias is b/2 because logits
        are summed. The pooled descriptor has no influence at step 0, so
        sigmoid(b) is the gate value at the first forward pass. With b=4 the
        gate starts at ~0.982, making sigmoid CGM behave near identity at
        training start. The model then learns to deviate downward only when
        the gradient signal supports it. Has no effect when init_bias <= 0
        (preserves the original random init for backward compatibility).

        Must be called AFTER the global trunc_normal init (FasterNet does this
        in its __init__ after self.apply(self._init_weights)), otherwise the
        global init will overwrite it.
        """
        if self.init_bias <= 0.0:
            return
        effective_bias = self.init_bias / 2.0 if self.pooling == "gap_gmp" else self.init_bias
        if self.gate_type == "eca":
            nn.init.zeros_(self.gate.weight)
            if self.gate.bias is not None:
                nn.init.constant_(self.gate.bias, effective_bias)
        else:
            last_conv = self.gate[-1]
            nn.init.zeros_(last_conv.weight)
            if last_conv.bias is not None:
                nn.init.constant_(last_conv.bias, effective_bias)

    def _gate_logits(self, pooled: Tensor) -> Tensor:
        if self.gate_type == "eca":
            return self.gate(pooled.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return self.gate(pooled)

    def forward(self, x: Tensor) -> Tensor:
        logits = self._gate_logits(self.avg_pool(x))
        if self.pooling == "gap_gmp":
            logits = logits + self._gate_logits(self.max_pool(x))
        gate = torch.sigmoid(logits)
        self.latest_gate = gate.detach()
        scale = 1.0 + self.alpha * (gate - 0.5) if self.mode == "residual" else gate
        self.latest_scale = scale.detach()
        return x * scale


class PartialConv3(nn.Module):
    """Partial 3x3 convolution with optional gating on processed channels."""

    def __init__(
        self,
        dim: int,
        n_div: int,
        forward_type: Literal["split_cat", "slicing"] = "split_cat",
        use_cgm: bool = False,
        cgm_reduction: int = 4,
        cgm_mode: CGMMode = "sigmoid",
        cgm_alpha: float = 0.5,
        cgm_type: CGMType = "se",
        eca_kernel_size: int = 3,
        cgm_pooling: CGMPooling = "gap",
        cgm_init_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.forward_type = forward_type
        self.partial_conv3 = nn.Conv2d(
            self.dim_conv3, self.dim_conv3, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.channel_gate = (
            ChannelGateModule(
                self.dim_conv3,
                reduction=cgm_reduction,
                mode=cgm_mode,
                alpha=cgm_alpha,
                gate_type=cgm_type,
                eca_kernel_size=eca_kernel_size,
                pooling=cgm_pooling,
                init_bias=cgm_init_bias,
            )
            if use_cgm
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.forward_type == "split_cat":
            return self.forward_split_cat(x)
        if self.forward_type == "slicing":
            return self.forward_slicing(x)
        raise ValueError(f"Unsupported PConv forward type: {self.forward_type}")

    def forward_slicing(self, x: Tensor) -> Tensor:
        x = x.clone()
        x1 = self.partial_conv3(x[:, : self.dim_conv3, :, :])
        x[:, : self.dim_conv3, :, :] = self.channel_gate(x1)
        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.channel_gate(self.partial_conv3(x1))
        return torch.cat((x1, x2), dim=1)


class MLPBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_div: int,
        mlp_ratio: float,
        drop_path: float,
        layer_scale_init_value: float,
        act_layer: type[nn.Module],
        norm_layer: type[nn.Module],
        pconv_fw_type: Literal["split_cat", "slicing"],
        use_cgm: bool,
        cgm_reduction: int,
        cgm_mode: CGMMode,
        cgm_alpha: float,
        cgm_type: CGMType,
        eca_kernel_size: int,
        cgm_pooling: CGMPooling,
        cgm_init_bias: float,
    ) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.spatial_mixing = PartialConv3(
            dim=dim,
            n_div=n_div,
            forward_type=pconv_fw_type,
            use_cgm=use_cgm,
            cgm_reduction=cgm_reduction,
            cgm_mode=cgm_mode,
            cgm_alpha=cgm_alpha,
            cgm_type=cgm_type,
            eca_kernel_size=eca_kernel_size,
            cgm_pooling=cgm_pooling,
            cgm_init_bias=cgm_init_bias,
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            norm_layer(hidden_dim),
            act_layer(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_scale = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = self.mlp(x)
        if self.layer_scale is not None:
            x = self.layer_scale.view(1, -1, 1, 1) * x
        return shortcut + self.drop_path(x)


class BasicStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        n_div: int,
        mlp_ratio: float,
        drop_path: Iterable[float],
        layer_scale_init_value: float,
        norm_layer: type[nn.Module],
        act_layer: type[nn.Module],
        pconv_fw_type: Literal["split_cat", "slicing"],
        use_cgm: bool,
        cgm_reduction: int,
        cgm_mode: CGMMode,
        cgm_alpha: float,
        cgm_type: CGMType,
        eca_kernel_size: int,
        cgm_pooling: CGMPooling,
        cgm_init_bias: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                MLPBlock(
                    dim=dim,
                    n_div=n_div,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path_i,
                    layer_scale_init_value=layer_scale_init_value,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    pconv_fw_type=pconv_fw_type,
                    use_cgm=use_cgm,
                    cgm_reduction=cgm_reduction,
                    cgm_mode=cgm_mode,
                    cgm_alpha=cgm_alpha,
                    cgm_type=cgm_type,
                    eca_kernel_size=eca_kernel_size,
                    cgm_pooling=cgm_pooling,
                    cgm_init_bias=cgm_init_bias,
                )
                for drop_path_i in drop_path
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int,
        patch_stride: int,
        in_chans: int,
        embed_dim: int,
        norm_layer: Optional[type[nn.Module]],
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, bias=False
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.proj(x))


class PatchMerging(nn.Module):
    def __init__(
        self,
        patch_size: int,
        patch_stride: int,
        dim: int,
        norm_layer: type[nn.Module],
    ) -> None:
        super().__init__()
        self.reduction = nn.Conv2d(
            dim, 2 * dim, kernel_size=patch_size, stride=patch_stride, bias=False
        )
        self.norm = norm_layer(2 * dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.reduction(x))


def _use_cgm_for_stage(stage_idx: int, placement: CGMPlacement) -> bool:
    if placement == "none":
        return False
    if placement == "all":
        return True
    if placement == "early":
        return stage_idx in {0, 1}
    if placement == "late":
        return stage_idx in {2, 3}
    if placement == "s2s3":
        return stage_idx in {1, 2}
    if placement == "s2s4":
        return stage_idx in {1, 3}
    if placement in {"s1", "s2", "s3", "s4"}:
        return stage_idx == int(placement[1]) - 1
    raise ValueError(f"Unknown CGM placement: {placement}")


class FasterNet(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 100,
        embed_dim: int = 40,
        depths: tuple[int, int, int, int] = (1, 2, 8, 2),
        mlp_ratio: float = 2.0,
        n_div: int = 4,
        patch_size: int = 4,
        patch_stride: int = 4,
        patch_size2: int = 2,
        patch_stride2: int = 2,
        patch_norm: bool = True,
        feature_dim: int = 1280,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 0.0,
        norm_layer: Literal["BN"] = "BN",
        act_layer: Literal["RELU", "GELU"] = "GELU",
        pconv_fw_type: Literal["split_cat", "slicing"] = "split_cat",
        cgm_placement: CGMPlacement = "none",
        cgm_reduction: int = 4,
        cgm_mode: CGMMode = "sigmoid",
        cgm_alpha: float = 0.5,
        cgm_type: CGMType = "se",
        eca_kernel_size: int = 3,
        cgm_pooling: CGMPooling = "gap",
        cgm_init_bias: float = 0.0,
    ) -> None:
        super().__init__()
        if norm_layer != "BN":
            raise ValueError("This CIFAR implementation currently supports BatchNorm only.")
        norm = nn.BatchNorm2d
        act = nn.GELU if act_layer == "GELU" else partial(nn.ReLU, inplace=True)

        self.num_classes = num_classes
        self.depths = depths
        self.cgm_placement = cgm_placement
        self.cgm_mode = cgm_mode
        self.cgm_alpha = cgm_alpha
        self.cgm_type = cgm_type
        self.eca_kernel_size = eca_kernel_size
        self.cgm_pooling = cgm_pooling
        self.cgm_init_bias = float(cgm_init_bias)
        self.num_features = int(embed_dim * 2 ** (len(depths) - 1))

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            patch_stride=patch_stride,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm if patch_norm else None,
        )

        drop_rates: List[float] = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        stages: list[nn.Module] = []
        for stage_idx, depth in enumerate(depths):
            dim = int(embed_dim * 2**stage_idx)
            start = sum(depths[:stage_idx])
            end = sum(depths[: stage_idx + 1])
            stages.append(
                BasicStage(
                    dim=dim,
                    depth=depth,
                    n_div=n_div,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_rates[start:end],
                    layer_scale_init_value=layer_scale_init_value,
                    norm_layer=norm,
                    act_layer=act,
                    pconv_fw_type=pconv_fw_type,
                    use_cgm=_use_cgm_for_stage(stage_idx, cgm_placement),
                    cgm_reduction=cgm_reduction,
                    cgm_mode=cgm_mode,
                    cgm_alpha=cgm_alpha,
                    cgm_type=cgm_type,
                    eca_kernel_size=eca_kernel_size,
                    cgm_pooling=cgm_pooling,
                    cgm_init_bias=cgm_init_bias,
                )
            )
            if stage_idx < len(depths) - 1:
                stages.append(
                    PatchMerging(
                        patch_size=patch_size2,
                        patch_stride=patch_stride2,
                        dim=dim,
                        norm_layer=norm,
                    )
                )
        self.stages = nn.Sequential(*stages)
        self.avgpool_pre_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.num_features, feature_dim, kernel_size=1, bias=False),
            act(),
        )
        self.head = nn.Linear(feature_dim, num_classes)
        self.apply(self._init_weights)
        # CGM identity init must run AFTER the global trunc_normal init above,
        # otherwise self.apply(self._init_weights) would overwrite it.
        for module in self.modules():
            if isinstance(module, ChannelGateModule):
                module.identity_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        x = self.stages(x)
        x = self.avgpool_pre_head(x)
        x = torch.flatten(x, 1)
        return self.head(x)

    @torch.no_grad()
    def collect_gate_means(self) -> dict[str, float]:
        stats: dict[str, float] = {}
        for name, module in self.named_modules():
            if isinstance(module, ChannelGateModule) and module.latest_gate is not None:
                stats[name] = float(module.latest_gate.mean().cpu())
        return stats

    @torch.no_grad()
    def collect_scale_means(self) -> dict[str, float]:
        stats: dict[str, float] = {}
        for name, module in self.named_modules():
            if isinstance(module, ChannelGateModule) and module.latest_scale is not None:
                stats[name] = float(module.latest_scale.mean().cpu())
        return stats


@dataclass(frozen=True)
class FasterNetConfig:
    embed_dim: int
    depths: tuple[int, int, int, int]
    drop_path_rate: float


MODEL_CONFIGS = {
    "fasternet_t0": FasterNetConfig(embed_dim=40, depths=(1, 2, 8, 2), drop_path_rate=0.0),
    "fasternet_t1": FasterNetConfig(embed_dim=64, depths=(1, 2, 8, 2), drop_path_rate=0.02),
}


def build_fasternet(
    model_name: Literal["fasternet_t0", "fasternet_t1"],
    num_classes: int,
    image_size: int,
    cgm_placement: CGMPlacement = "none",
    cgm_reduction: int = 4,
    cgm_mode: CGMMode = "sigmoid",
    cgm_alpha: float = 0.5,
    cgm_type: CGMType = "se",
    eca_kernel_size: int = 3,
    cgm_pooling: CGMPooling = "gap",
    cgm_init_bias: float = 0.0,
) -> FasterNet:
    cfg = MODEL_CONFIGS[model_name]
    patch_size = 2 if image_size <= 64 else 4
    return FasterNet(
        num_classes=num_classes,
        embed_dim=cfg.embed_dim,
        depths=cfg.depths,
        drop_path_rate=cfg.drop_path_rate,
        patch_size=patch_size,
        patch_stride=patch_size,
        cgm_placement=cgm_placement,
        cgm_reduction=cgm_reduction,
        cgm_mode=cgm_mode,
        cgm_alpha=cgm_alpha,
        cgm_type=cgm_type,
        eca_kernel_size=eca_kernel_size,
        cgm_pooling=cgm_pooling,
        cgm_init_bias=cgm_init_bias,
    )


def fasternet_t0(
    num_classes: int = 100,
    image_size: int = 32,
    cgm_placement: CGMPlacement = "none",
    cgm_reduction: int = 4,
    cgm_mode: CGMMode = "sigmoid",
    cgm_alpha: float = 0.5,
    cgm_type: CGMType = "se",
    eca_kernel_size: int = 3,
    cgm_pooling: CGMPooling = "gap",
    cgm_init_bias: float = 0.0,
) -> FasterNet:
    return build_fasternet(
        "fasternet_t0",
        num_classes,
        image_size,
        cgm_placement,
        cgm_reduction,
        cgm_mode,
        cgm_alpha,
        cgm_type,
        eca_kernel_size,
        cgm_pooling,
        cgm_init_bias,
    )


def fasternet_t1(
    num_classes: int = 100,
    image_size: int = 32,
    cgm_placement: CGMPlacement = "none",
    cgm_reduction: int = 4,
    cgm_mode: CGMMode = "sigmoid",
    cgm_alpha: float = 0.5,
    cgm_type: CGMType = "se",
    eca_kernel_size: int = 3,
    cgm_pooling: CGMPooling = "gap",
    cgm_init_bias: float = 0.0,
) -> FasterNet:
    return build_fasternet(
        "fasternet_t1",
        num_classes,
        image_size,
        cgm_placement,
        cgm_reduction,
        cgm_mode,
        cgm_alpha,
        cgm_type,
        eca_kernel_size,
        cgm_pooling,
        cgm_init_bias,
    )
