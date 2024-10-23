import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
print(sys.executable)
from functools import partial, wraps

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn

from aroma.fourier_features import MultiScaleNeRFEncoding, NeRFEncoding
# ACKNOWLEDGEMENT: code adapted from perceiver implementation by lucidrains


# 0. utils function
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


def dropout_seq(images, coordinates, mask=None, dropout=0.25):
    b, n, *_, device = *images.shape, images.device
    logits = torch.randn(b, n, device=device)

    keep_prob = 1.0 - dropout
    num_keep = max(1, int(keep_prob * n))
    keep_indices = logits.topk(num_keep, dim=1).indices

    batch_indices = torch.arange(b, device=device)
    batch_indices = rearrange(batch_indices, "b -> b 1")

    if mask is None:
        images = images[batch_indices, keep_indices]
        coordinates = coordinates[batch_indices, keep_indices]

        return images, coordinates

    else:
        images = images[batch_indices, keep_indices]
        coordinates = coordinates[batch_indices, keep_indices]
        mask = mask[batch_indices, keep_indices]

        return images, coordinates, mask


class DiagonalGaussianDistribution(object):
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean
        self.logvar = logvar
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.mean.device
            )

    def sample(self, K=1):
        if K == 1:
            x = self.mean + self.std * torch.randn(self.mean.shape).to(
                device=self.mean.device
            )
            return x
        else:
            x = self.mean[None, ...].repeat([K, 1, 1, 1]) + self.std[None, ...].repeat(
                K, 1, 1, 1
            ) * torch.randn([K, *self.mean.shape]).to(device=self.mean.device)
            return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.mean(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2]
                )
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2],
                )

    def nll(self, sample, dims=[1, 2]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean


def linear_scheduler(start, end, num_steps):
    delta = (end - start) / num_steps
    return [start + i * delta for i in range(num_steps)]


# 1. Main blocks


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class PreNormCross(nn.Module):
    def __init__(self, dim, fn, k_dim=None, v_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(k_dim) if exists(k_dim) else None
        self.norm_v = nn.LayerNorm(v_dim) if exists(v_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_v):
            k = kwargs["k"]
            v = kwargs["v"]
            normed_k = self.norm_k(k)
            normed_v = self.norm_v(v)
            kwargs.update(k=normed_k, v=normed_v)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, use_geglu=False):
        super().__init__()
        if use_geglu:
            self.net = nn.Sequential(
                nn.Linear(dim, dim * mult * 2),
                GEGLU(),
                nn.Linear(dim * mult, dim),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, dim * mult),
                nn.GELU(),
                nn.Linear(dim * mult, dim),
            )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, heads=8, dim_head=64, dropout=0):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(key_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(value_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.attn_drop = nn.Dropout(dropout, inplace=False)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, k, v, mask=None, pos=None):
        h = self.heads

        q = self.to_q(x)
        # context = default(context, x)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = mask.bool()
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            # mask = repeat(mask, "b j -> (b h) () j", h=h)
            mask = repeat(mask, "b j -> (b h) (n) j", h=h, n=x.shape[1])
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.resid_drop(self.to_out(out))


class MultiScaleAttention(nn.Module):
    def __init__(
        self, query_dim, context_dim=None, out_dim=None, heads=8, dim_head=64, dropout=0
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        out_dim = default(out_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, out_dim)

        self.attn_drop = nn.Dropout(dropout, inplace=False)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, pos=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (k, v))

        # the s stands for scale as we have queries at different frequency bandwidths.
        q = rearrange(q, "b n s (h d) -> (b h) s n d", h=h)
        sim = einsum("b s i d, b j d -> b s i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = einsum("b s i j, b j d -> b s i d", attn, v)
        out = rearrange(out, "(b h) s n d -> b n s (h d)", h=h)
        return self.resid_drop(self.to_out(out))


class Attention(nn.Module):
    def __init__(
        self, query_dim, context_dim=None, out_dim=None, heads=8, dim_head=64, dropout=0
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        out_dim = default(out_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, out_dim)

        self.attn_drop = nn.Dropout(dropout, inplace=False)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, pos=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention
        attn = sim.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.resid_drop(self.to_out(out))


# 2. INR Decoder


class FourierPositionalEmbedding(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        num_freq=32,
        max_freq_log2=5,
        input_dim=2,
        base_freq=2,
        use_relu=True,
    ):
        super().__init__()

        self.nerf_embedder = NeRFEncoding(
            num_freq=num_freq,
            max_freq_log2=max_freq_log2,
            input_dim=input_dim,
            base_freq=base_freq,
            log_sampling=False,
            include_input=False,
        )

        self.linear = nn.Linear(self.nerf_embedder.out_dim, hidden_dim)
        self.use_relu = use_relu

    def forward(self, coords):
        x = self.nerf_embedder(coords)
        if self.use_relu:
            x = torch.relu(self.linear(x))
        else:
            x = self.linear(x)  # try without relu

        return x


class LocalityAwareINRDecoder(nn.Module):
    def __init__(self, output_dim=1, embed_dim=16, num_scales=3, dim=128, depth=3):
        super().__init__()
        self.dim = dim
        # Define Fourier transformation, linear layers, and other components
        self.depth = depth
        layers = [nn.Linear(embed_dim * num_scales, dim), nn.ReLU()]  # Input layer

        # Add intermediate layers based on depth
        for _ in range(depth - 1):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(dim, output_dim))  # Output layer
        self.mlp = nn.Sequential(*layers)

    def forward(self, localized_latents):
        # we stack the different scales
        localized_latents = einops.rearrange(localized_latents, "b n s c -> b n (s c)")
        return self.mlp(localized_latents)


class AdaLN(nn.Module):
    def __init__(self, hidden_dim):
        super(AdaLN, self).__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc_scale = nn.Linear(hidden_dim, hidden_dim)
        self.fc_shift = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, z):
        # Apply LayerNorm first
        x_ln = self.ln(x)
        # Compute scale and shift parameters conditioned on z
        scale = self.fc_scale(z)  # .unsqueeze(1)
        shift = self.fc_shift(z)  # .unsqueeze(1)
        # Apply AdaLN transformation
        return scale * x_ln + shift


# Residual block class
class ModulationBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ModulationBlock, self).__init__()
        self.adaln1 = AdaLN(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.silu = nn.SiLU()
        self.adaln2 = AdaLN(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, z):
        # Apply first AdaLN and linear transformation
        residual = x
        out = self.adaln1(x, z)
        out = self.silu(self.linear1(out))
        # Apply second AdaLN and linear transformation
        out = self.adaln2(out, z)
        out = self.linear2(out)
        # Residual connection
        return out + residual


# not used  but looks interesting
class LocalityAwareINRDecoderWithModulation(nn.Module):
    def __init__(self, hidden_dim=256, num_blocks=2):
        super(LocalityAwareINRDecoderWithModulation, self).__init__()
        # Stack residual blocks
        self.blocks = nn.ModuleList(
            [ModulationBlock(hidden_dim) for _ in range(num_blocks)]
        )

    def forward(self, x, z):
        # Pass through each residual block
        for block in self.blocks:
            x = block(x, z)
        return x


# 3. Perceiver Encoder
class PerceiverEncoder(nn.Module):
    def __init__(
        self,
        *,
        num_channels=1,
        input_dim=2,
        depth=3,
        num_latents=64,
        hidden_dim=64,
        latent_dim=16,
        mlp_feature_dim=16,
        cross_heads=8,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        max_pos_encoding_freq=4,
        num_freq=12,
        scales=[3, 4, 5],
        bottleneck_index=0,
        decoder_ff=False,
        encode_geo=False,
        include_pos_in_value=False,
    ):
        super().__init__()
        self.depth = depth
        self.bottleneck_index = bottleneck_index  # where to put the botleneck, by default 0 means just after cross attention
        self.encode_geo = encode_geo
        self.include_pos_in_value=include_pos_in_value

        if include_pos_in_value:
            self.pos_encoding = FourierPositionalEmbedding(hidden_dim=hidden_dim,
                                                          num_freq=num_freq,
                                                          max_freq_log2=max_pos_encoding_freq,
                                                          input_dim=input_dim,
                                                          base_freq=2,
                                                          use_relu=True)
        else:
            self.pos_encoding = NeRFEncoding(
                num_freq=num_freq,
                max_freq_log2=max_pos_encoding_freq,
                input_dim=input_dim,
                base_freq=2,
                log_sampling=True,
                include_input=False,
                use_pi=True,
            )

        self.pos_query = MultiScaleNeRFEncoding(
            num_freq,
            log_sampling=True,
            include_input=False,
            input_dim=input_dim,
            base_freq=2,
            scales=scales,
            use_pi=True,
            disjoint=True,
        )  # False

        small_std = False
        sigma = 0.02 if small_std else 1
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_dim) * sigma)

        value_dim = hidden_dim
        if include_pos_in_value:
            key_dim = hidden_dim
        else:
            key_dim = self.pos_encoding.out_dim
        queries_dim = self.pos_query.out_dim_per_scale
        self.lift_values = nn.Linear(num_channels, hidden_dim)

        # Cross attend the pixels
        self.cross_attend_blocks = nn.ModuleList(
            [
                PreNormCross(
                    hidden_dim,
                    CrossAttention(
                        hidden_dim,
                        key_dim,
                        value_dim,
                        heads=cross_heads,
                        dim_head=cross_dim_head,
                    ),
                    k_dim=key_dim,
                    v_dim=value_dim,
                ),
                PreNorm(hidden_dim, FeedForward(hidden_dim)),
            ]
        )
        # cross attend the coordinates
        if self.encode_geo:
            self.cross_attend_geo = nn.ModuleList(
                [
                    PreNormCross(
                        hidden_dim,
                        CrossAttention(
                            hidden_dim,
                            key_dim,
                            key_dim,
                            heads=cross_heads,
                            dim_head=cross_dim_head,
                        ),
                        k_dim=key_dim,
                        v_dim=key_dim,
                    ),
                    PreNorm(hidden_dim, FeedForward(hidden_dim)),
                ]
            )
        get_latent_attn = lambda: PreNorm(
            hidden_dim,
            Attention(hidden_dim, heads=latent_heads, dim_head=latent_dim_head),
        )
        get_latent_ff = lambda: PreNorm(hidden_dim, FeedForward(hidden_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])

        for i in range(depth):
            self.layers.append(nn.ModuleList([get_latent_attn(), get_latent_ff()]))

        self.decoder_cross_attn = PreNorm(
            queries_dim,
            MultiScaleAttention(
                queries_dim,
                hidden_dim,
                mlp_feature_dim,
                heads=cross_heads,
                dim_head=cross_dim_head,
            ),
            context_dim=hidden_dim,
        )
        self.decoder_ff = (
            PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        )

        self.mean_fc = nn.Linear(hidden_dim, latent_dim)
        self.logvar_fc = nn.Linear(hidden_dim, latent_dim)
        self.lift_z = nn.Linear(latent_dim, hidden_dim)

    def forward(
        self,
        images,
        coords,
        mask=None,
        target_coords=None,
        sample_posterior=True,
        return_stats=False,
    ):
        b, *_, device = *images.shape, images.device

        if target_coords is None:
            queries = self.pos_query(coords)
        else:
            queries = self.pos_query(target_coords)

        x = repeat(self.latents, "n d -> b n d", b=b)

        k = self.pos_encoding(coords)
        v = self.lift_values(images)

        # if encode_geo, cross attend to the other pixel locations
        if self.encode_geo:
            cross_attn, cross_ff = self.cross_attend_geo
            x = cross_attn(x, k=k, v=k, mask=mask) + x
            x = cross_ff(x) + x

        # cross attend the coordinates
        cross_attn, cross_ff = self.cross_attend_blocks

        x = cross_attn(x, k=k, v=v+k if self.include_pos_in_value else v, mask=mask) + x
        x = cross_ff(x) + x

        # layers
        for index, (self_attn, self_ff) in enumerate(self.layers):
            if index == self.bottleneck_index:
                # bottleneck
                mu = self.mean_fc(x)
                logvar = self.logvar_fc(x)
                posterior = DiagonalGaussianDistribution(mu, logvar)

                if sample_posterior:
                    z = posterior.sample()
                else:
                    z = posterior.mode()

                x = self.lift_z(z)

            x = self_attn(x) + x
            x = self_ff(x) + x

        if self.bottleneck_index == len(self.layers):
            # bottleneck
            mu = self.mean_fc(x)
            logvar = self.logvar_fc(x)
            posterior = DiagonalGaussianDistribution(mu, logvar)

            if sample_posterior:
                z = posterior.sample()
            else:
                z = posterior.mode()

            x = self.lift_z(z)

        # cross attend from decoder queries to latents
        latents = self.decoder_cross_attn(queries, context=x)

        # optional decoder feedforward

        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)

        # final linear out
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        if return_stats:
            return latents, kl_loss, mu, logvar

        return latents, kl_loss

    def get_features(self, images, coords, mask=None):
        b, *_, device = *images.shape, images.device

        c = coords.clone()

        x = repeat(self.latents, "n d -> b n d", b=b)

        k = self.pos_encoding(c)
        v = self.lift_values(images)

        # if encode_geo, cross attend to the other pixel locations
        if self.encode_geo:
            cross_attn, cross_ff = self.cross_attend_geo
            x = cross_attn(x, k=k, v=k, mask=mask) + x
            x = cross_ff(x) + x

        cross_attn, cross_ff = self.cross_attend_blocks

        # cross attention only happens once for Perceiver IO
        x = cross_attn(x, k=k, v=v+k if self.include_pos_in_value else v, mask=mask) + x
        x = cross_ff(x) + x

        # layers

        for index, (self_attn, self_ff) in enumerate(self.layers):
            if index == self.bottleneck_index:
                mu = self.mean_fc(x)
                logvar = self.logvar_fc(x)
                # posterior = DiagonalGaussianDistribution(mu, logvar)

                return mu, logvar
            x = self_attn(x) + x
            x = self_ff(x) + x

        if self.bottleneck_index == len(self.layers):
            # bottleneck
            mu = self.mean_fc(x)
            logvar = self.logvar_fc(x)
            return mu, logvar
           

    def process(self, features, coords):
        queries = self.pos_query(coords)
        x = features

        # cross attend from decoder queries to latents
        for index, (self_attn, self_ff) in enumerate(self.layers):
            if self.bottleneck_index == index:
                x = self.lift_z(features)
                x = self_attn(x) + x
                x = self_ff(x) + x
            elif self.bottleneck_index > index:
                pass

            else:
                x = self_attn(x) + x
                x = self_ff(x) + x

        if self.bottleneck_index == len(self.layers):
            x = self.lift_z(features)

        latents = self.decoder_cross_attn(queries, context=x)

        # optional decoder feedforward
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)

        # final linear out
        return latents

    def process_from_stats(self, mean, logvar, queries):
        posterior = DiagonalGaussianDistribution(mean, logvar)
        z = posterior.sample()
        x = self.lift_z(z)

        # layers

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        if not exists(queries):
            return x

        # make sure queries contains batch dimension

        if queries.ndim == 2:
            queries = repeat(queries, "n d -> b n d", b=b)

        # cross attend from decoder queries to latents

        latents = self.decoder_cross_attn(queries, context=x)

        # optional decoder feedforward

        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)

        # final linear out

        return latents, z

    def process_from_codes(self, codes, queries):
        b = codes.shape[0]
        z = codes
        x = self.lift_z(z)

        # layers

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        if not exists(queries):
            return x

        # make sure queries contains batch dimension

        if queries.ndim == 2:
            queries = repeat(queries, "n d -> b n d", b=b)

        # cross attend from decoder queries to latents
        latents = self.decoder_cross_attn(queries, context=x)

        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)

        return latents, z


class AROMAEncoderDecoderKL(nn.Module):
    def __init__(
        self,
        input_dim=2,
        num_channels=1,
        hidden_dim=64,
        dim=64,
        num_self_attentions=3,
        num_latents=16,
        latent_dim=8,  
        latent_heads=12,
        latent_dim_head=64,
        cross_heads=8,
        cross_dim_head=64,
        scales=[3, 4, 5],
        mlp_feature_dim=16,
        depth_inr=3,
        bottleneck_index=0,
        max_pos_encoding_freq=4,
        num_freq=12,
        encode_geo=False,
        include_pos_in_value=False
    ):
        super().__init__()

        self.encoder = PerceiverEncoder(
            num_channels=num_channels,  # num input channels
            depth=num_self_attentions,  # depth of net
            num_latents=num_latents,  # num tokens used for cross attention
            hidden_dim=hidden_dim,  # dimensions of tokens
            latent_dim=latent_dim,  # latent dimension of the reduced representation
            cross_heads=cross_heads,  # number of heads for cross attention.
            latent_heads=latent_heads,  # number of heads for latent self attention
            cross_dim_head=cross_dim_head,  # number of dimensions per cross attention head
            latent_dim_head=latent_dim_head,  # number of dimensions per latent self attention head
            input_dim=input_dim,  # dimension of the coordinate: 1-> 1D, 2-> 2D.
            max_pos_encoding_freq=max_pos_encoding_freq,  # maximum frequency embedding for encoding the pixels
            scales=scales,  # scales used for decoding
            mlp_feature_dim=mlp_feature_dim,  # feature dimensions before the MLP
            bottleneck_index=bottleneck_index,  # index of the bottleneck layer
            encode_geo=encode_geo,  # whether to encode implicitly the geometry
            num_freq=num_freq,  # number of frequencies for the positional embedding
            include_pos_in_value=include_pos_in_value, # whether to mix the first pixel location and pixel value in the first CA.
        )

        self.decoder = LocalityAwareINRDecoder(
            output_dim=num_channels,
            embed_dim=mlp_feature_dim,
            num_scales=len(scales),
            dim=dim,
            depth=depth_inr,
        )

    def forward(
        self,
        images,
        coords,
        mask=None,
        target_coords=None,
        return_stats=False,
        sample_posterior=True,
    ):
        if return_stats:
            localized_latents, kl_loss, mean, logvar = self.encoder(
                images,
                coords,
                mask,
                target_coords,
                return_stats=return_stats,
                sample_posterior=sample_posterior,
            )
        else:
            localized_latents, kl_loss = self.encoder(
                images,
                coords,
                mask,
                target_coords,
                return_stats=return_stats,
                sample_posterior=sample_posterior,
            )

        output_features = self.decoder(localized_latents)

        if return_stats:
            return output_features, kl_loss, mean, logvar

        return output_features, kl_loss

    def encode(self, images, coords, mask=None):
        mu, logvar = self.encoder.get_features(images, coords, mask)

        return mu, logvar

    def decode(self, features, coords):
        localized_latents = self.encoder.process(features, coords)
        output_features = self.decoder(localized_latents)

        return output_features
