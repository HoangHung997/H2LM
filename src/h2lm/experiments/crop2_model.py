"""Experimental single-line CNN + visual cross-attention decoder, all weights learned together."""
from __future__ import annotations

import torch
from torch import nn

from .data import VOCAB


class CropVLM(nn.Module):
    def __init__(self, d_model: int = 96, height: int = 48, width: int = 320):
        super().__init__()
        if (any(type(v) is not int for v in (d_model, height, width))
                or not 32 <= d_model <= 128 or d_model % 4 or not 32 <= height <= 64
                or height % 16 or not 128 <= width <= 384 or width % 4):
            raise ValueError('Invalid bounded crop model configuration')
        self.height, self.width = height, width
        channels = [1, 16, 32, 64, d_model]
        strides = [2, 2, (2,1), (2,1)]
        blocks = []
        for a, b, stride in zip(channels, channels[1:], strides):
            blocks += [nn.Conv2d(a,b,3,stride,1), nn.GELU()]
        self.vision = nn.Sequential(*blocks)
        self.vpos = nn.Parameter(torch.randn(1,width//4,d_model)*0.02)
        self.vnorm = nn.LayerNorm(d_model)
        layer = nn.TransformerEncoderLayer(d_model,4,d_model*3,dropout=0,batch_first=True,norm_first=True)
        self.visual_context = nn.TransformerEncoder(layer,1,enable_nested_tensor=False)
        self.embedding = nn.Embedding(VOCAB,d_model,padding_idx=3)
        self.tpos = nn.Parameter(torch.randn(1,64,d_model)*0.02)
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(d_model,4,d_model*3,dropout=0,
                                     batch_first=True,norm_first=True) for _ in range(2)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model,VOCAB)
        self.ctc_head = nn.Linear(d_model,VOCAB)  # Auxiliary only. Never decoded/passed to language.
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.zeros_(module.in_proj_bias)

    def memory(self, pixels: torch.Tensor) -> torch.Tensor:
        if (pixels.ndim != 4 or pixels.shape[1:] != (1,self.height,self.width)
                or not 1 <= pixels.shape[0] <= 32 or not torch.isfinite(pixels).all()):
            raise ValueError('Expected configured grayscale crop dimensions')
        x = self.vision(pixels).mean(2).transpose(1,2)
        return self.visual_context(self.vnorm(x) + self.vpos)

    def decode(self, ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= 64 or ids.shape[0] != memory.shape[0]:
            raise ValueError('Invalid token input')
        x = self.embedding(ids) + self.tpos[:,:ids.shape[1]]
        mask = torch.triu(torch.ones(ids.shape[1],ids.shape[1],device=ids.device,dtype=torch.bool),1)
        for layer in self.layers:
            x = layer(x,memory,tgt_mask=mask,tgt_key_padding_mask=ids.eq(3))
        return self.head(self.norm(x))

    def forward(self, ids: torch.Tensor, pixels: torch.Tensor):
        memory = self.memory(pixels)
        return self.decode(ids,memory), self.ctc_head(memory)
