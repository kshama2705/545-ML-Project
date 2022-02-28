from typing import Optional

import torch
from torch import nn


class PreNormTransformerEncoderLayer(nn.TransformerEncoderLayer):
    r"""
    A variant of :class:`torch.nn.TransformerEncoderLayer` where layer
    normalization is included inside the residual branch, and performed before
    self-attention and feedforward layers.

    Refer documentation of :class:`torch.nn.TransformerEncoderLayer` for more
    details on the API.
    """

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # fmt: off
        # We use the members (modules) from super-class, just the order of
        # operations is changed here. First layernorm, then attention.
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)

        # Layernorm first, then transformation through feedforward network.
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class PreNormTransformerDecoderLayer(nn.TransformerDecoderLayer):
    r"""
    A variant of :class:`torch.nn.TransformerDecoderLayer` where layer
    normalization is included inside the residual branch, and performed before
    self-attention and feedforward layers.

    Refer documentation of :class:`torch.nn.TransformerDecoderLayer` for more
    details on the API.
    """

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # fmt: off
        # We use the members (modules) from super-class, just the order of
        # operations is changed here. First layernorm, then attention.
        tgt2 = self.norm1(tgt)
        tgt2, _ = self.self_attn(
            tgt2, tgt2, tgt2, attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)

        # Layernorm first, then decoder attention.
        tgt2 = self.norm2(tgt)
        tgt2, _ = self.multihead_attn(
            tgt2, memory, memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)

        # Layernorm first, then transformation through feedforward network.
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt
