from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float

from .modules import GRN, DropPath
from .norm import CF, AdaLayerNorm


def ConvND(
    spatial_dims: int,
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    output_padding=0,  # only for ConvTransposeND
    dilation=1,
    groups=1,
    bias=True,
    padding_mode="zeros",
    transpose=False,
    device=None,
    dtype=None,
):
    assert spatial_dims in [1, 2, 3], "dims must be 1, 2, or 3"
    conv = getattr(nn, f"Conv{'Transpose' if transpose else ''}{spatial_dims}d")
    if padding == "auto":
        padding = kernel_size // 2
    args = dict(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        padding_mode=padding_mode,
        device=device,
        dtype=dtype,
    )
    if transpose:
        args["output_padding"] = output_padding
    return conv(**args)


class UpsampleND(nn.Module):
    """An upsampling layer."""

    def __init__(
        self,
        dims: int,
        in_channels: int,
        out_channels: Optional[int] = None,
        use_conv_transpose: bool = False,
        kernel_size: Optional[int] = None,
    ):
        super().__init__()
        self.dims = dims
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv_transpose = use_conv_transpose

        self.norm = nn.LayerNorm(in_channels)

        if use_conv_transpose:
            if kernel_size is None:
                kernel_size = 4
            self.conv = ConvND(
                self.dims,
                self.in_channels,
                self.out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding="auto",
                transpose=True,
            )
        else:
            if kernel_size is None:
                kernel_size = 3
            self.conv = ConvND(
                self.dims,
                self.in_channels,
                self.out_channels,
                kernel_size=kernel_size,
                padding="auto",
            )

        self.interp_mode = (
            "linear" if self.dims == 1 else "bicubic" if self.dims == 2 else "trilinear"
        )

    def forward(
        self,
        x: Float[torch.Tensor, "b c ..."],
    ) -> Float[torch.Tensor, "b c ..."]:
        x = rearrange(x, "b c ... -> b ... c").contiguous()
        x = self.norm(x)
        x = rearrange(x, "b ... c -> b c ...").contiguous()

        if self.use_conv_transpose:
            return self.conv(x)
        else:
            x = F.interpolate(x, scale_factor=2.0, mode=self.interp_mode)
            x = self.conv(x)

        return x


class DownsampleND(nn.Module):
    def __init__(
        self,
        dims: int,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.dims = dims
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels

        self.norm = nn.LayerNorm(in_channels)

        self.conv = ConvND(
            self.dims,
            self.in_channels,
            self.out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding="auto",
        )

    def forward(
        self,
        x: Float[torch.Tensor, "b c ..."],
    ) -> Float[torch.Tensor, "b c ..."]:
        x = rearrange(x, "b c ... -> b ... c").contiguous()
        x = self.norm(x)
        x = rearrange(x, "b ... c -> b c ...").contiguous()

        x = self.conv(x)
        return x


class ResNetBlockND(nn.Module):
    r"""
    A ResNet block with optionally conditional normalization.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        conditional_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        non_linearity (`str`, *optional*, default to `"gelu"`): the activation function to use.
        up (`bool`, *optional*, default to `False`): If `True`, add an upsample layer.
        down (`bool`, *optional*, default to `False`): If `True`, add a downsample layer.
    """

    def __init__(
        self,
        spatial_dims,
        in_channels: int,
        out_channels: int,
        conditional_channels: int = 0,
        dropout: float = 0.0,
        shortcut: bool = True,
        kernel_size: int = 3,
        up: bool = False,
        down: bool = False,
        init_bias: float = 0.0,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.shortcut = shortcut
        self.kernel_size = kernel_size
        self.up = up
        self.down = down
        self.conditional_channels = conditional_channels

        if self.conditional_channels == 0:
            self.register_buffer("fake_t_emb", torch.zeros(1, 1))

        self.norm1 = AdaLayerNorm(in_channels, max(conditional_channels, 1))

        self.conv1 = ConvND(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding="auto",
        )

        self.norm2 = AdaLayerNorm(out_channels, max(conditional_channels, 1))

        self.dropout = torch.nn.Dropout(dropout)

        self.conv2 = ConvND(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding="auto",
        )
        nn.init.zeros_(self.conv2.bias)

        self.upsample = self.downsample = None

        if self.up:
            self.upsample = UpsampleND(
                spatial_dims, in_channels, kernel_size=kernel_size
            )
        elif self.down:
            self.downsample = DownsampleND(
                spatial_dims, in_channels, kernel_size=kernel_size
            )

        self.conv_shortcut = (
            ConvND(
                spatial_dims,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            if shortcut
            else nn.Identity()
        )
        nn.init.constant_(self.conv_shortcut.bias, init_bias)

    def forward(
        self,
        x: Float[torch.Tensor, "b c ..."],
        t_emb: Optional[Float[torch.Tensor, "b n"]] = None,
    ) -> Float[torch.Tensor, "b c ..."]:
        out = x

        if t_emb is None:
            assert self.conditional_channels == 0
            t_emb = self.fake_t_emb.to(x)

        out = rearrange(out, "b c ... -> b ... c").contiguous()
        out = self.norm1(out, t_emb)
        out = rearrange(out, "b ... c -> b c ...").contiguous()

        out = F.gelu(out)

        if self.upsample is not None:
            x = self.upsample(x)
            out = self.upsample(out)

        elif self.downsample is not None:
            x = self.downsample(x)
            out = self.downsample(out)

        out = self.conv1(out)

        out = rearrange(out, "b c ... -> b ... c").contiguous()
        out = self.norm2(out, t_emb)
        out = rearrange(out, "b ... c -> b c ...").contiguous()

        out = F.gelu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        return self.conv_shortcut(x) + out


class ResNetEncoderND(nn.Sequential):
    def __init__(
        self,
        spatial_dims,
        input_channels,
        output_channels,
        kernel_sizes=(3, 3, 3),
        num_layers=3,
        **kwargs,
    ):
        # super().__init__(
        #     ResNetBlockND(dims, input_channels, output_channels // 4, down=True),
        #     ResNetBlockND(dims, output_channels // 4, output_channels // 2, down=True),
        #     ResNetBlockND(dims, output_channels // 2, output_channels, down=True),
        # )
        in_channels = [input_channels] + [
            output_channels // (2 ** (i)) for i in reversed(range(1, num_layers))
        ]
        out_channels = [
            output_channels // (2 ** (i)) for i in reversed(range(num_layers))
        ]
        assert len(kernel_sizes) == num_layers
        super().__init__(
            *[
                ResNetBlockND(
                    spatial_dims,
                    max(in_channels[i], 1),
                    max(out_channels[i], 1),
                    kernel_size=kernel_sizes[i],
                    down=True,
                    **kwargs,
                )
                for i in range(num_layers)
            ]
        )
        self.spatial_dims = spatial_dims


class ResNetDecoderND(nn.Sequential):
    def __init__(
        self,
        spatial_dims,
        input_channels,
        output_channels,
        kernel_sizes=(3, 3, 3),
        num_layers=3,
        **kwargs,
    ):
        # super().__init__(
        #     ResNetBlockND(dims, input_channels, input_channels // 2, up=True),
        #     ResNetBlockND(dims, input_channels // 2, input_channels // 4, up=True),
        #     ResNetBlockND(dims, input_channels // 4, output_channels, up=True),
        # )
        in_channels = [input_channels // (2 ** (i)) for i in range(num_layers)]
        out_channels = [input_channels // (2 ** (i)) for i in range(1, num_layers)] + [
            output_channels
        ]
        assert len(kernel_sizes) == num_layers
        super().__init__(
            *[
                ResNetBlockND(
                    spatial_dims,
                    max(in_channels[i], 1),
                    max(out_channels[i], 1),
                    kernel_size=kernel_sizes[i],
                    up=True,
                    **kwargs,
                )
                for i in range(num_layers)
            ]
        )
        self.spatial_dims = spatial_dims


class ConvNextBlockND(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self, spatial_dims, embed_dim, drop_path=0.0, layer_scale_init_value=1e-6
    ):
        super().__init__()
        self.dwconv = ConvND(
            spatial_dims=spatial_dims,
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=7,
            padding=3,
            groups=embed_dim,
        )  # depthwise conv
        self.norm = nn.LayerNorm(embed_dim)
        self.pwconv1 = nn.Linear(
            embed_dim, 4 * embed_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.SiLU(inplace=True)
        self.pwconv2 = nn.Linear(4 * embed_dim, embed_dim)
        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((embed_dim)), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = rearrange(x, "b c ... -> b ... c").contiguous()
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = rearrange(x, "b ... c -> b c ...").contiguous()
        x = input + self.drop_path(x)
        return x


class ConvNextV2BlockND(nn.Module):
    """ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, spatial_dims, embed_dim, drop_path=0.0):
        super().__init__()
        self.spatial_dims = (spatial_dims,)
        self.dwconv = ConvND(
            spatial_dims=spatial_dims,
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=7,
            padding=3,
            groups=embed_dim,
        )  # depthwise conv
        self.norm = nn.LayerNorm(embed_dim)
        self.pwconv1 = nn.Linear(
            embed_dim, 4 * embed_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.SiLU(inplace=True)
        self.grn = GRN(spatial_dims=spatial_dims, embed_dim=4 * embed_dim)
        self.pwconv2 = nn.Linear(4 * embed_dim, embed_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = rearrange(x, "b c ... -> b ... c").contiguous()
        # x = x.permute(0, 2, 1)  # (B, C, N) -> (B, N, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = rearrange(x, "b ... c -> b c ...").contiguous()
        # x = x.permute(0, 2, 1)  # (B, N, C) -> (B, C, N)
        x = input + self.drop_path(x)
        return x


class ConvNextEncoderND(nn.Module):
    r"""ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        spatial_dims,
        in_chans=2,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        v2=False,
    ):
        super().__init__()
        assert len(depths) == len(dims), "depths and dims should have the same length"
        self.spatial_dims = spatial_dims
        self.depths = depths
        self.dims = dims
        num_layers = len(depths)
        convnext_block = (
            ConvNextV2BlockND
            if v2
            else partial(ConvNextBlockND, layer_scale_init_value=layer_scale_init_value)
        )

        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            ConvND(
                spatial_dims=spatial_dims,
                in_channels=in_chans,
                out_channels=dims[0],
                kernel_size=2,
                stride=2,
            ),
            CF(nn.LayerNorm(dims[0])),
        )
        self.downsample_layers.append(stem)
        for i in range(num_layers - 1):
            downsample_layer = nn.Sequential(
                CF(nn.LayerNorm(dims[i])),
                ConvND(
                    spatial_dims=spatial_dims,
                    in_channels=dims[i],
                    out_channels=dims[i + 1],
                    kernel_size=2,
                    stride=2,
                ),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        cur = 0
        for i in range(num_layers):
            stage = nn.Sequential(
                *[
                    convnext_block(
                        spatial_dims=spatial_dims,
                        embed_dim=dims[i],
                        drop_path=dp_rates[cur + j],
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = CF(nn.LayerNorm(dims[-1]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for ds, st in zip(self.downsample_layers, self.stages):
            x = ds(x)
            x = st(x)
        return self.norm(x)


class ConvNextDecoderND(nn.Module):
    r"""ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        spatial_dims,
        in_chans=768,
        depths=[3, 3, 9, 3],
        dims=[384, 192, 96, 2],
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        v2=False,
    ):
        super().__init__()
        assert len(depths) == len(dims), "depths and dims should have the same length"
        self.spatial_dims = spatial_dims
        self.depths = depths
        self.dims = dims
        num_layers = len(depths)
        convnext_block = (
            ConvNextV2BlockND
            if v2
            else partial(ConvNextBlockND, layer_scale_init_value=layer_scale_init_value)
        )

        self.upsample_layers = nn.ModuleList()

        stem = nn.Sequential(
            ConvND(
                spatial_dims,
                in_channels=in_chans,
                out_channels=dims[0],
                kernel_size=2,
                stride=2,
                transpose=True,
            ),
            CF(nn.LayerNorm(dims[0])),
        )
        self.upsample_layers.append(stem)

        for i in range(num_layers - 1):
            upsample_layer = nn.Sequential(
                CF(nn.LayerNorm(dims[i])),
                ConvND(
                    spatial_dims,
                    in_channels=dims[i],
                    out_channels=dims[i + 1],
                    kernel_size=2,
                    stride=2,
                    transpose=True,
                ),
            )
            self.upsample_layers.append(upsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        cur = 0
        for i in range(num_layers):
            stage = nn.Sequential(
                *[
                    convnext_block(
                        spatial_dims=spatial_dims,
                        embed_dim=dims[i],
                        drop_path=dp_rates[cur + j],
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for us, st in zip(self.upsample_layers, self.stages):
            x = us(x)
            x = st(x)
        return x
