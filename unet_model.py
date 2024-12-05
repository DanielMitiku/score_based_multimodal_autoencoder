import math
from inspect import isfunction
from functools import partial
from einops import rearrange
import torch
from torch import nn, einsum
import numpy as np
import torch.nn.functional as F

# Taken and updated from https://huggingface.co/blog/annotated-diffusion
#UNET-SM

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""
    
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        h = self.block2(h)
        return h + self.res_conv(x)
    
class ConvNextBlock(nn.Module):
    """https://arxiv.org/abs/2201.03545"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if exists(self.mlp) and exists(time_emb):
            assert exists(time_emb), "time embedding must be passed in"
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")

        h = self.net(h)
        return h + self.res_conv(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)
        self.dim_mults = dim_mults

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time=None):
        pad_dim_1, pad_dim_2 = 0, 0
        if 2**np.ceil(np.log2(x.shape[-1])) - x.shape[-1] != 0:
            pad_dim_1 = int((2**np.ceil(np.log2(x.shape[-1])) - x.shape[-1]) // 2)
            x = F.pad(x, (pad_dim_1, pad_dim_1),"constant",0)
            # print('pad_dim1: ', pad_dim_1)
        if 2**np.ceil(np.log2(x.shape[-2])) - x.shape[-2] != 0:
            pad_dim_2 = int((2**np.ceil(np.log2(x.shape[-2])) - x.shape[-2]) // 2)
            x = F.pad(x, (0, 0, pad_dim_2, pad_dim_2),"constant",0)
            # print('pad_dim2: ', pad_dim_2)


        x = self.init_conv(x)
        # print('init conv', x.shape)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
            # print('x down: ', x.shape)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        # print('mid: ', x.shape)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
            # print('x up: ', x.shape)
        
        x = self.final_conv(x)
        if pad_dim_1 != 0:
            x = x[..., pad_dim_1:-pad_dim_1]
        if pad_dim_2 != 0:
            x = x[..., pad_dim_2:-pad_dim_2, :]
        return x

class UnetEnergy(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
        size_z=64,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)
        self.dim_mults = dim_mults

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )

        self.e_out = nn.Sequential(
            nn.Softplus(),
            nn.Linear(out_dim*size_z,1),
            )

    def forward(self, x, time=None):
        pad_dim_1, pad_dim_2 = 0, 0
        if 2**np.ceil(np.log2(x.shape[-1])) - x.shape[-1] != 0:
            pad_dim_1 = int((2**np.ceil(np.log2(x.shape[-1])) - x.shape[-1]) // 2)
            x = F.pad(x, (pad_dim_1, pad_dim_1),"constant",0)
            # print('pad_dim1: ', pad_dim_1)
        if 2**np.ceil(np.log2(x.shape[-2])) - x.shape[-2] != 0:
            pad_dim_2 = int((2**np.ceil(np.log2(x.shape[-2])) - x.shape[-2]) // 2)
            x = F.pad(x, (0, 0, pad_dim_2, pad_dim_2),"constant",0)
            # print('pad_dim2: ', pad_dim_2)


        x = self.init_conv(x)
        # print('init conv', x.shape)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
            # print('x down: ', x.shape)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        # print('mid: ', x.shape)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
            # print('x up: ', x.shape)
        
        x = self.final_conv(x)
        if pad_dim_1 != 0:
            x = x[..., pad_dim_1:-pad_dim_1]
        if pad_dim_2 != 0:
            x = x[..., pad_dim_2:-pad_dim_2, :]
        energy = self.e_out(x.view(x.shape[0],-1))
        return x, energy

class UnetNodown(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)
        self.dim_mults = dim_mults

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        # Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        # Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time=None):
        pad_dim_1, pad_dim_2 = 0, 0
        if 2**np.ceil(np.log2(x.shape[-1])) - x.shape[-1] != 0:
            pad_dim_1 = int((2**np.ceil(np.log2(x.shape[-1])) - x.shape[-1]) // 2)
            x = F.pad(x, (pad_dim_1, pad_dim_1),"constant",0)
            # print('pad_dim1: ', pad_dim_1)
        if 2**np.ceil(np.log2(x.shape[-2])) - x.shape[-2] != 0:
            pad_dim_2 = int((2**np.ceil(np.log2(x.shape[-2])) - x.shape[-2]) // 2)
            x = F.pad(x, (0, 0, pad_dim_2, pad_dim_2),"constant",0)
            # print('pad_dim2: ', pad_dim_2)


        x = self.init_conv(x)
        # print('init conv', x.shape)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # downsample
        for block1, block2, attn in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            # x = downsample(x)
            # print('x down: ', x.shape)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        # print('mid: ', x.shape)

        # upsample
        for block1, block2, attn in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            # x = upsample(x)
            # print('x up: ', x.shape)
        
        x = self.final_conv(x)
        if pad_dim_1 != 0:
            x = x[..., pad_dim_1:-pad_dim_1]
        if pad_dim_2 != 0:
            x = x[..., pad_dim_2:-pad_dim_2, :]
        return x

class Lin(nn.Module):
    def __init__(self,init_size):
        super().__init__()

        self.init_size = init_size
        self.lin_layers = nn.Sequential(
            nn.Linear(self.init_size, self.init_size*2),
            nn.Linear(self.init_size*2, self.init_size*2),
            nn.Linear(self.init_size*2, self.init_size),
        )

    def forward(self,x):
        return self.lin_layers(x)

class Lin2(nn.Module):
    def __init__(self,init_size, out_size):
        super().__init__()

        self.init_size = init_size
        self.out_size = out_size
        self.lin_layers = nn.Sequential(
            nn.Linear(self.init_size, self.init_size*2),
            # nn.LayerNorm(self.init_size*2),
            nn.ReLU(),
            nn.Linear(self.init_size*2, self.init_size*2),
            # nn.LayerNorm(self.init_size*4),
            nn.ReLU(),
            nn.Linear(self.init_size*2, self.out_size),
        )

    def forward(self,x):
        return self.lin_layers(x)

class Lin3(nn.Module):
    def __init__(self,init_size, out_size):
        super().__init__()

        self.init_size = init_size
        self.out_size = out_size
        self.lin_layers = nn.Sequential(
            nn.Linear(self.init_size, self.init_size*2),
            nn.ReLU(),
            nn.Linear(self.init_size*2, self.init_size*2),
            nn.ReLU(),
            nn.Linear(self.init_size*2, self.out_size),
        )

    def forward(self,x):
        return x + self.lin_layers(x)


class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def scaled_dot_product(self, q, k, v, mask=None):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values

    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values = self.scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        return o

class NorUnet(nn.Module):
    def __init__(self, 
        n_mod,
        z_dim,
        dim,
        dim2,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 2, 2),
        channels=3,
        with_time_emb=False,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
        cross=True
    ):
        super().__init__()
        
        self.n_mod = n_mod
        self.dim = dim
        self.dim2 = dim2
        self.z_dim = z_dim
        # self.cross = cross
        # self.init_lin = Lin(self.z_dim*self.n_mod)
        # self.cross_att = MultiheadAttention(input_dim=self.z_dim, embed_dim=self.z_dim, num_heads=4)
        # self.mask = torch.ones(self.n_mod, self.n_mod) - torch.eye(self.n_mod)
        # self.lin2conv = nn.Linear(self.z_dim, self.z_dim)
        self.unet = Unet(dim, init_dim, out_dim, dim_mults, channels, with_time_emb, resnet_block_groups, use_convnext, convnext_mult)


    def forward(self, x, sigma):
        x = self.unet(x)
        return x / sigma

class CAUNET(nn.Module):
    def __init__(self, 
        n_mod,
        z_dim,
        dim,
        dim2,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 2, 2),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
        cross=True
    ):
        super().__init__()
        
        self.n_mod = n_mod
        self.dim = dim
        self.dim2 = dim2
        self.z_dim = z_dim
        self.cross = cross
        self.init_lin = Lin(self.z_dim*self.n_mod)
        self.cross_att = MultiheadAttention(input_dim=self.z_dim, embed_dim=self.z_dim, num_heads=4)
        self.mask = torch.ones(self.n_mod, self.n_mod) - torch.eye(self.n_mod)
        self.lin2conv = nn.Linear(self.z_dim, self.z_dim)
        self.unet = Unet(dim, init_dim, out_dim, dim_mults, channels, with_time_emb, resnet_block_groups, use_convnext, convnext_mult)


    def forward(self, x, time=None):
        x = self.init_lin(x.view(-1,self.z_dim*self.n_mod)).view(-1,self.n_mod,self.z_dim)
        if self.cross:
            x = self.cross_att(x, self.mask.to(x.device))
        else:
            x = self.cross_att(x)
        x = self.lin2conv(x).view(-1, self.n_mod, self.dim, self.dim2)
        x = self.unet(x, time).view(-1, self.n_mod, self.z_dim)
        return x

def scaled_dot_product(q, k, v, mask=None):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values

class MHA(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)

    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)
        return q, k, v

class CRA(nn.Module):
    def __init__(self, 
        n_mod,
        z_dim,
        dim,
        dim2,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 2, 2),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
        cross=True
    ):
        super().__init__()
        
        self.n_mod = n_mod
        self.dim = dim
        self.dim2 = dim2
        self.z_dim = z_dim
        self.cross = cross
        self.clin1 = Lin3(self.z_dim, self.z_dim)
        self.clin2 = Lin3(self.z_dim, self.z_dim)
        self.qkv1 = MHA(input_dim=self.z_dim, embed_dim=self.z_dim, num_heads=4)
        self.qkv2 = MHA(input_dim=self.z_dim, embed_dim=self.z_dim, num_heads=4)
        self.o_proj = nn.Linear(self.z_dim, self.z_dim)

        self.mask = torch.ones(self.n_mod, self.n_mod) - torch.eye(self.n_mod)
        self.lin2conv = nn.Linear(self.z_dim, self.z_dim)
        self.unet = Unet(dim, init_dim, out_dim, dim_mults, channels, with_time_emb, resnet_block_groups, use_convnext, convnext_mult)


    def forward(self, x, time=None):
        x1 = self.clin1(x)
        x2 = self.clin2(x)

        q1, _, _ = self.qkv1(x1)
        _, k2, v2 = self.qkv2(x2)

        values = scaled_dot_product(q1, k2, v2)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(x.shape[0], -1, self.z_dim)
        values = self.o_proj(values).view(x.shape[0],-1,self.dim, self.dim2)
        out = self.unet(values, time).view(-1, self.n_mod, self.z_dim)
        return out


class Lincat(nn.Module):
    def __init__(self,z_dim, n_mod, mask):
        super().__init__()

        self.z_dim = z_dim
        self.n_mod = n_mod
        self.mask = mask
        self.lin= Lin2(self.z_dim, self.z_dim)
        self.n1 = nn.LayerNorm(self.z_dim)
        self.n2 = nn.LayerNorm(self.z_dim)
        # self.time_lin = nn.Linear(1,self.z_dim*self.n_mod)
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(self.z_dim),
                nn.Linear(self.z_dim, self.z_dim),
                nn.GELU(),
                nn.Linear(self.z_dim, self.z_dim),
            )
        self.cross_att = MultiheadAttention(input_dim=self.z_dim, embed_dim=self.z_dim, num_heads=4)

    def forward(self,x, time):
        x = self.n1( x + self.cross_att(x, self.mask.to(x.device)))
        x = self.n2(x + self.lin(x))
        time = self.time_mlp(time)
        x = x + time.unsqueeze(1)
        return x


class CAUNET2(nn.Module):
    def __init__(self, 
        n_mod,
        z_dim,
        dim,
        dim2,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 2, 2),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
        n_block=3,
        cross=True
    ):
        super().__init__()
        
        self.n_mod = n_mod
        self.dim = dim
        self.dim2 = dim2
        self.z_dim = z_dim
        self.cross = cross
        self.n_block = n_block
        if self.cross:
            self.mask = torch.ones(self.n_mod, self.n_mod) - torch.eye(self.n_mod)
        else:
            self.mask = torch.ones(self.n_mod, self.n_mod)
        self.c_att = nn.ModuleList([Lincat(self.z_dim, self.n_mod, self.mask) for i in range(self.n_block)])
        self.lin2conv = nn.Linear(self.z_dim, self.z_dim)
        self.unet = Unet(dim, init_dim, out_dim, dim_mults, channels, with_time_emb, resnet_block_groups, use_convnext, convnext_mult)


    def forward(self, x, time):
        for m in self.c_att:
            x = m(x, time)
        # x = self.lin2conv(x).view(x.shape[0], self.n_mod, self.dim, self.dim2)
        x = self.unet(x.view(x.shape[0], self.n_mod, self.dim, self.dim2), time)
        return x.view(-1, self.n_mod, self.z_dim)

class CAUNET3(nn.Module):
    def __init__(self, 
        n_mod,
        z_dim,
        dim,
        dim2,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 2, 2),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
        cross=True
    ):
        super().__init__()
        
        self.n_mod = n_mod
        self.dim = dim
        self.dim2 = dim2
        self.z_dim = z_dim
        self.cross = cross
        self.init_lin = nn.Linear(self.z_dim*self.n_mod, self.z_dim*self.n_mod)
        self.cross_att = MultiheadAttention(input_dim=self.z_dim, embed_dim=self.z_dim, num_heads=4)
        self.mask = torch.ones(self.n_mod, self.n_mod) - torch.eye(self.n_mod)
        self.lin2conv = nn.Linear(self.z_dim, self.z_dim)
        self.unet = Unet(dim, init_dim, out_dim, dim_mults, channels, with_time_emb, resnet_block_groups, use_convnext, convnext_mult)


    def forward(self, x, time=None):
        x = self.init_lin(x.view(-1,self.z_dim*self.n_mod)).view(-1,self.n_mod,self.z_dim)
        if self.cross:
            x = x + self.cross_att(x, self.mask.to(x.device))
        else:
            x = x + self.cross_att(x)
        x = self.lin2conv(x).view(-1, self.n_mod, self.dim, self.dim2)
        x = self.unet(x, time).view(-1, self.n_mod, self.z_dim)
        return x

class CAUNET4(nn.Module):
    def __init__(self, 
        n_mod,
        z_dim,
        dim,
        dim2,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 2, 2),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
        cross=True
    ):
        super().__init__()
        
        self.n_mod = n_mod
        self.dim = dim
        self.dim2 = dim2
        self.z_dim = z_dim
        self.cross = cross
        self.init_lin = nn.Linear(self.z_dim*self.n_mod, self.z_dim*self.n_mod)
        self.cross_att = MultiheadAttention(input_dim=self.z_dim, embed_dim=self.z_dim, num_heads=4)
        self.mask = torch.ones(self.n_mod, self.n_mod) - torch.eye(self.n_mod)
        self.lin2conv = nn.Linear(self.z_dim, self.z_dim)
        self.unet = Unet(dim, init_dim, out_dim, dim_mults, channels, with_time_emb, resnet_block_groups, use_convnext, convnext_mult)


    def forward(self, x, time=None):
        x = self.init_lin(x.view(-1,self.z_dim*self.n_mod)).view(-1,self.n_mod,self.z_dim)
        if self.cross:
            x = x + self.cross_att(x, self.mask.to(x.device))
        else:
            x = x + self.cross_att(x)
        x = self.lin2conv(x).view(-1, self.n_mod, self.dim, self.dim2)
        x = self.unet(x, time).view(-1, self.n_mod, self.z_dim)
        return x


class UnetZ(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
        z_dim=256,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.z_dim = z_dim
        self.z_mlp = nn.Sequential(
                nn.Linear(self.z_dim, dims[-1]),
                nn.GELU(),
                nn.Linear(dims[-1], dims[-1]),
            )
        
        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, z, time=None):
        x = self.init_conv(x)
        # print('init conv', x.shape)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
            # print('x down: ', x.shape)

        # bottleneck
        x = self.mid_block1(x, t)
        # Add z in the bottleneck
        z = self.z_mlp(z).view(-1,x.shape[1],1,1)
        x = x + z
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        x = x + z
        # print('mid: ', x.shape)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
            # print('x up: ', x.shape)

        return self.final_conv(x)


class UnetVAE(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(self.channels * 2, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, x_hat, time=None):
        x = self.init_conv(torch.cat([x,x_hat],dim=1))
        # print('init conv', x.shape)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
            # print('x down: ', x.shape)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        # print('mid: ', x.shape)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
            # print('x up: ', x.shape)

        return self.final_conv(x)


class SM3simple(nn.Module):

    def __init__(self, n_mod=3, size_z=1024):
        super().__init__()
        self.size_z = size_z
        self.n_mod = n_mod
  
        self.layers = nn.Sequential(
            nn.Linear(self.size_z*self.n_mod, self.size_z*2),
            nn.Softplus(),
            nn.Linear(self.size_z*2, self.size_z*2),
            nn.Softplus(),
            nn.Linear(self.size_z*2, self.size_z*2),
            nn.Softplus(),
            nn.Linear(self.size_z*2, self.size_z),
            nn.Softplus(),
            nn.Linear(self.size_z, self.size_z*2),
            nn.Softplus(),
            nn.Linear(self.size_z*2, self.size_z*self.n_mod),)

    def forward(self, x, sigma):
        return self.layers(x) / sigma