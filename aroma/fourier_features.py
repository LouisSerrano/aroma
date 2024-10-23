from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

# from aroma.siren import LatentToModulation
# from aroma.utils.film_conditioning import film, film_linear, film_translate


class GaussianEncoding(nn.Module):
    def __init__(self, embedding_size, scale, dims=2, gaussian=True):
        super().__init__()
        self.embedding_size = embedding_size
        self.scale = scale

        if gaussian:
            bvals = torch.randn(embedding_size // 2, dims) * scale
        else:
            bvals = 2.0 ** torch.linspace(0, scale, embedding_size // 2) - 1

            if dims == 1:
                bvals = bvals[:, None]

            elif dims == 2:
                bvals = torch.stack([bvals, torch.zeros_like(bvals)], dim=-1)
                bvals = torch.cat([bvals, torch.roll(bvals, 1, -1)], dim=0)

            else:
                tmp = (dims - 1) * (torch.zeros_like(bvals),)
                bvals = torch.stack([bvals, *tmp], dim=-1)

                tmp = [torch.roll(bvals, i, -1) for i in range(1, dims)]
                bvals = torch.cat([bvals, *tmp], dim=0)

        avals = torch.ones((bvals.shape[0]))
        self.avals = nn.Parameter(avals, requires_grad=False)
        self.bvals = nn.Parameter(bvals, requires_grad=False)

    def forward(self, tensor) -> torch.Tensor:
        """
        Apply positional encoding to the input.
        """

        return torch.cat(
            [
                self.avals * torch.sin((2.0 * np.pi * tensor) @ self.bvals.T),
                self.avals * torch.cos((2.0 * np.pi * tensor) @ self.bvals.T),
            ],
            dim=-1,
        )


class MultiScaleNeRFEncoding(nn.Module):
    def __init__(
        self,
        num_freqs_per_scale,
        log_sampling=True,
        include_input=False,
        input_dim=3,
        base_freq=2,
        scales=[3, 4, 5],
        use_pi=True,
        disjoint=True,
    ):
        super().__init__()

        self.num_freqs_per_scale = num_freqs_per_scale
        self.log_sampling = log_sampling
        self.include_input = include_input
        self.input_dim = input_dim
        self.base_freq = base_freq
        self.scales = scales
        self.use_pi = use_pi
        self.disjoint = disjoint

        # Initialize bands for each scale
        self.bands = nn.ParameterDict(self.initialize_bands(scales))

        # Calculate output dimension
        self.out_dim = 0
        if include_input:
            self.out_dim += input_dim
        for scale in scales:
            self.out_dim += (
                num_freqs_per_scale * input_dim * 2
            )  # sin and cos for each frequency band

        self.out_dim_per_scale = num_freqs_per_scale * input_dim * 2

    def initialize_bands(self, scales):
        s = [0] + scales
        bands = {}
        for j in range(len(s) - 1):
            if self.disjoint:
                start, end = s[j], s[j + 1]
            else:
                start, end = s[0], s[j + 1]

            if self.log_sampling:
                band = self.base_freq ** torch.linspace(
                    start, end, steps=self.num_freqs_per_scale
                )
            else:
                band = torch.linspace(
                    self.base_freq**start,
                    self.base_freq**end,
                    steps=self.num_freqs_per_scale,
                )
            if self.use_pi:
                band = band * np.pi
            bands[f"{s[j+1]}"] = nn.Parameter(band, requires_grad=False)

        return bands

    def forward(self, coords):
        encoded_list = []

        if self.include_input:
            encoded_list.append(coords)

        for scale, bands in self.bands.items():
            b = coords.shape[0]
            N = coords.shape[1]
            winded = (coords[..., None] * bands[None, :]).reshape(b, N, -1)
            # .reshape(coords.shape[0], -1)
            encoded_scale = torch.cat([torch.sin(winded), torch.cos(winded)], dim=-1)
            encoded_list.append(encoded_scale)
            # print('encoded_scale', encoded_scale.shape)

        return torch.stack(encoded_list, dim=-2)

    def name(self) -> str:
        return "Multiscale Positional Encoding"

    def public_properties(self) -> Dict[str, Any]:
        return {
            "Output Dim": self.out_dim,
            "Num. Frequencies": self.num_freqs_per_scale * len(self.scales),
            "Max Frequency": f"2^{self.max_freq_log2}",
            "Include Input": self.include_input,
            "Scales": self.scales,
        }


class NeRFEncoding(nn.Module):
    """PyTorch implementation of regular positional embedding, as used in the original NeRF and Transformer papers."""

    def __init__(
        self,
        num_freq,
        max_freq_log2,
        log_sampling=True,
        include_input=True,
        input_dim=3,
        base_freq=2,
        use_pi=True,
    ):
        """Initialize the module.
        Args:
            num_freq (int): The number of frequency bands to sample.
            max_freq_log2 (int): The maximum frequency.
                                 The bands will be sampled at regular intervals in [0, 2^max_freq_log2].
            log_sampling (bool): If true, will sample frequency bands in log space.
            include_input (bool): If true, will concatenate the input.
            input_dim (int): The dimension of the input coordinate space.
        Returns:
            (void): Initializes the encoding.
        """
        super().__init__()

        self.num_freq = num_freq
        self.max_freq_log2 = max_freq_log2
        self.log_sampling = log_sampling
        self.include_input = include_input
        self.out_dim = 0
        self.base_freq = base_freq
        self.use_pi = use_pi

        if include_input:
            self.out_dim += input_dim

        if self.log_sampling:
            self.bands = self.base_freq ** torch.linspace(
                0.0, max_freq_log2, steps=num_freq
            )
            if use_pi:
                self.bands = self.bands * np.pi
        else:
            self.bands = torch.linspace(
                1, self.base_freq**max_freq_log2, steps=num_freq
            )
            if use_pi:
                self.bands = self.bands * np.pi

        # The out_dim is really just input_dim + num_freq * input_dim * 2 (for sin and cos)
        self.out_dim += self.bands.shape[0] * input_dim * 2
        self.bands = nn.Parameter(self.bands).requires_grad_(False)

    def forward(self, coords, with_batch=True):
        """Embeds the coordinates.
        Args:
            coords (torch.FloatTensor): Coordinates of shape [N, input_dim]
        Returns:
            (torch.FloatTensor): Embeddings of shape [N, input_dim + out_dim] or [N, out_dim].
        """
        if with_batch:
            N = coords.shape[0]
            winded = (coords[..., None, :] * self.bands[None, None, :, None]).reshape(
                N, coords.shape[1], coords.shape[-1] * self.num_freq
            )
            encoded = torch.cat([torch.sin(winded), torch.cos(winded)], dim=-1)
            if self.include_input:
                encoded = torch.cat([coords, encoded], dim=-1)

        else:
            N = coords.shape[0]
            winded = (coords[:, None] * self.bands[None, :, None]).reshape(
                N, coords.shape[1] * self.num_freq
            )
            encoded = torch.cat([torch.sin(winded), torch.cos(winded)], dim=-1)
            if self.include_input:
                encoded = torch.cat([coords, encoded], dim=-1)
        return encoded

    def name(self) -> str:
        """A human readable name for the given wisp module."""
        return "Positional Encoding"

    def public_properties(self) -> Dict[str, Any]:
        """Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.
        """
        return {
            "Output Dim": self.out_dim,
            "Num. Frequencies": self.num_freq,
            "Max Frequency": f"2^{self.max_freq_log2}",
            "Include Input": self.include_input,
        }


if __name__ == "__main__":
    multi_scale_nerf = MultiScaleNeRFEncoding(
        8, 5, include_input=False, input_dim=2, disjoint=False
    )
    print(multi_scale_nerf.bands["3"])
    print(multi_scale_nerf.bands["4"])
    print(multi_scale_nerf.bands["5"])
    print(multi_scale_nerf.out_dim)
    x = torch.Tensor([0.1, 0.2]).unsqueeze(0)
    y = multi_scale_nerf(x)
    print("y", y.shape)
