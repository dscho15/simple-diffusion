import torch
import torch.nn as nn
from torch.nn import Module
from einops import rearrange


class CrossAttention(Module):

    def __init__(
        self,
        dim1,
        dim2,
        *,
        num_heads,
        latent_dim=None,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()

        self.num_heads = num_heads
        latent_dim = latent_dim if latent_dim is not None else dim1
        head_dim = latent_dim // num_heads

        assert head_dim * num_heads == latent_dim, "dim must be divisible by num_heads"

        self.q = nn.Linear(dim1, latent_dim, bias=qkv_bias)
        self.kv = nn.Linear(dim2, latent_dim * 2, bias=qkv_bias)

        self.n_x = nn.LayerNorm(dim1)
        self.n_y = nn.LayerNorm(dim2)

        self.mha = nn.MultiheadAttention(
            latent_dim, num_heads, attn_drop, bias=False, batch_first=True
        )

        self.proj = nn.Linear(latent_dim, dim1)

    def forward(self, x, y):
        x = self.n_x(x)
        y = self.n_y(y)
        q = self.q(x)
        kv = self.kv(y)
        k, v = kv.chunk(2, dim=-1)

        h = self.mha(query=q, key=k, value=v)[0]
        return self.proj(h)


class FeedForward(Module):

    def __init__(self, dim, exp_factor=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * exp_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * exp_factor, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.net(self.norm(x))


class RinBlock(Module):

    def __init__(
        self,
        x_dim,
        num_heads,
        *,
        z_dim=None,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        ff_dropout=0.0,
    ):

        super().__init__()

        self.read_ca = CrossAttention(
            dim1=z_dim,
            dim2=x_dim,
            num_heads=num_heads,
            latent_dim=z_dim,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.read_ff = FeedForward(z_dim, exp_factor=4, dropout=ff_dropout)

        self.compute_sa = nn.ModuleList(
            [
                CrossAttention(
                    dim1=z_dim,
                    dim2=z_dim,
                    num_heads=num_heads,
                    latent_dim=z_dim,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                )
                for _ in range(2)
            ]
        )

        self.compute_ff = nn.ModuleList(
            [FeedForward(z_dim, exp_factor=4, dropout=ff_dropout) for _ in range(2)]
        )

        self.write_ca = CrossAttention(
            x_dim,
            z_dim,
            num_heads=num_heads,
            latent_dim=z_dim,
        )

        self.write_ff = FeedForward(x_dim, exp_factor=4, dropout=ff_dropout)


    def forward(self, x, z):

        z = self.read_ca(z, x) + z
        z = self.read_ff(z) + z

        for sa, ff in zip(self.compute_sa, self.compute_ff):
            z = sa(z, z) + z
            z = ff(z) + z

        x = self.write_ca(x, z) + x
        x = self.write_ff(x) + x

        return x
    

class LinearPatchEmbedder(Module):

    def __init__(self, dim_in, dim_out, patch_size = 4):
        super().__init__()
        self.patch_size = patch_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(dim_in * (patch_size ** 2), dim_out),
            torch.nn.LayerNorm(dim_out)
        )

    def forward(self, x):
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        x = self.model(x)
        return x
    
class LinearUnpatchEmbedder(Module):

    def __init__(self, dim_in, dim_out, patch_size = 4):
        super().__init__()
        self.patch_size = patch_size
        self.model = torch.nn.Sequential(
            torch.nn.LayerNorm(dim_in),
            torch.nn.Linear(dim_in, dim_out * (patch_size ** 2))
        )

    def forward(self, x, h, w):
        h = h // self.patch_size
        w = w // self.patch_size
        x = self.model(x)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size, h=h, w=w)
        return x

class Rin(Module):
    
    def __init__(self, x_dim, z_dim, num_blocks, num_heads, **kwargs):
        
        super().__init__()

        self.lpe = LinearPatchEmbedder(x_dim, z_dim)
        self.lue = LinearUnpatchEmbedder(z_dim, x_dim)

        self.rin_blocks = nn.ModuleList(
            [
                RinBlock(x_dim, num_heads, z_dim=z_dim, **kwargs)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):

        h, w = x.size()[-2:]

        x = self.lpe(x)

        z = torch.zeros_like(x)

        for rin_block in self.rin_blocks:
            x = rin_block(x, z)

        x = self.lue(x, h, w)

        return x