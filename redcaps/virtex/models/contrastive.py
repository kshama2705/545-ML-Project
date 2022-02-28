from typing import Any, Dict

import torch
from torch import nn
import torch.distributed as dist

from virtex.modules.label_smoothing import CrossEntropyLossWithLabelSmoothing
from virtex.modules.textual_heads import TextualHead
from virtex.modules.visual_backbones import VisualBackbone


class ImageTextContrastiveModel(nn.Module):
    def __init__(
        self,
        visual: VisualBackbone,
        textual: TextualHead,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.visual = visual
        self.textual = textual
        self.padding_idx = self.textual.padding_idx

        self.visual_projection = nn.Linear(
            self.visual.visual_feature_size,
            self.textual.textual_feature_size,
            bias=False,
        )
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/0.07)))
        self.loss = CrossEntropyLossWithLabelSmoothing(
            label_smoothing, ignore_index=self.padding_idx
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:

        # Check if logit_scale needs to be clipped from last iteration.
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 3.912)
        # 50 times

        # shape: (batch_size, channels, height, width)
        visual_features = self.visual(batch["image"])
        batch_size = visual_features.size(0)

        # shape: (batch_size, channels)
        visual_features = visual_features.mean(dim=[2, 3]).view(batch_size, -1)

        # shape: (batch_size, textual_feature_size)
        visual_features = self.visual_projection(visual_features)

        caption_tokens = batch["caption_tokens"]
        caption_lengths = batch["caption_lengths"]

        # shape: (batch_size, max_caption_length, hidden_size)
        textual_features = self.textual(caption_tokens, caption_lengths)

        # Take features from the first time-step (as BERT-* models do).
        # shape: (batch_size, hidden_size)
        textual_features = textual_features[:, 0, :]

        # Normalize visual and textual features.
        # shape: (batch_size, textual_feature_size)
        visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
        textual_features = textual_features / textual_features.norm(
            dim=-1, keepdim=True
        )
        # Gather textual features from all processes into one large tensor to
        # increase negative samples for contrastive learning.
        gathered_textual_features = [
            torch.zeros_like(textual_features) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered_textual_features, textual_features)

        # Shift features of current rank to zeroth index for easy implementation.
        gathered_textual_features[0], gathered_textual_features[dist.get_rank()] = (
            gathered_textual_features[dist.get_rank()],
            gathered_textual_features[0],
        )
        # shape: (batch_size * world_size, textual_feature_size)
        gathered_textual_features = torch.cat(gathered_textual_features, dim=0)

        # Calculate pairwise cosine similarity as logits.
        logit_scale = self.logit_scale.exp()
        visual_logits = logit_scale * visual_features @ gathered_textual_features.t()

        # Targets are an identity matrix (image [i] should match with caption [i])
        visual_loss = self.loss(
            visual_logits, torch.arange(visual_logits.size(0)).to(visual_logits.device)
        )

        # Do the same thing for visual features.
        gathered_visual_features = [
            torch.zeros_like(visual_features) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered_visual_features, visual_features)

        gathered_visual_features[0], gathered_visual_features[dist.get_rank()] = (
            gathered_visual_features[dist.get_rank()],
            gathered_visual_features[0],
        )
        # shape: (batch_size * world_size, textual_feature_size)
        gathered_visual_features = torch.cat(gathered_visual_features, dim=0)

        # Calculate pairwise cosine similarity as logits.
        logit_scale = self.logit_scale.exp()
        textual_logits = logit_scale * textual_features @ gathered_visual_features.t()

        # Targets are an identity matrix (image [i] should match with caption [i])
        textual_loss = self.loss(
            textual_logits,
            torch.arange(textual_logits.size(0)).to(textual_logits.device),
        )
        loss = 0.5 * (visual_loss + textual_loss)
        output_dict: Dict[str, Any] = {
            "loss": loss,
            # Single scalar per batch for logging in training script.
            "loss_components": {"contrastive": loss.clone().detach()},
        }

        return output_dict
