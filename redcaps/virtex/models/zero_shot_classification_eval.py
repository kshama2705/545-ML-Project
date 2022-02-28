import copy
import functools
from typing import Any, Dict

import json 

import torch
from torch import nn

from virtex.data.tokenizers import SentencePieceBPETokenizer
from virtex.modules.label_smoothing import CrossEntropyLossWithLabelSmoothing
from virtex.modules.textual_heads import TextualHead
from virtex.modules.visual_backbones import VisualBackbone


class ZeroShotClassifier(nn.Module):
    def __init__(
        self,
        visual: VisualBackbone,
        textual: TextualHead,
    ):
        super().__init__()
        self.visual = visual
        self.textual = textual
        self.padding_idx = self.textual.padding_idx

        # Clone the textual module for backward direction if doing captioning
        # in both directions (separately).
        self.backward_textual = copy.deepcopy(self.textual)

        # Share weights for visual projection, and input/output embeddings.
        self.backward_textual.visual_projection = self.textual.visual_projection
        self.backward_textual.embedding = self.textual.embedding
        self.backward_textual.output = self.textual.output
   
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_idx,reduction='none')

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        
        # shape: (batch_size, channels, height, width)
        visual_features = self.visual(batch["image"])
        batch_size = visual_features.size(0)
        
        classification_losses = []
          
        #catagories shape: (1000, 20)
        
        caption_tokens = batch["caption_tokens"]
        backward_caption_tokens = batch["noitpac_tokens"]
        caption_lengths = batch["caption_lengths"]
        print

        for i in range(caption_tokens.shape[0]):
            # shape : (batch size, 20)
            catagory_caption_tokens = caption_tokens[i,:].unsqueeze(0).repeat(batch_size,1)
            # shape : (batch size, 20)
            catagory_backward_caption_tokens = backward_caption_tokens[i,:].unsqueeze(0).repeat(batch_size,1)
            # shape : (batch size)
            catagory_caption_lengths = caption_lengths[i].unsqueeze(0).repeat(batch_size)
            
            #print("caption_tokens.shape:",caption_tokens.shape)
            #print("backward_caption_tokens.shape:",backward_caption_tokens.shape)
            #print("caption_lengths.shape:",caption_lengths.shape)
            
            #print("catagory_caption_tokens.shape:",catagory_caption_tokens.shape)
            #print("catagory_backward_caption_tokens.shape:",catagory_backward_caption_tokens.shape)
            #print("catagory_caption_lengths.shape:",catagory_caption_lengths.shape)
           
            output_logits = self.textual(
                visual_features, catagory_caption_tokens, catagory_caption_lengths
            )
            

            loss = self.loss(
                output_logits[:, :-1].contiguous().view(-1, self.textual.vocab_size),
                catagory_caption_tokens[:, 1:].contiguous().view(-1)
            )
            
            # Do captioning in backward direction if specified.
            backward_output_logits = self.backward_textual(
                visual_features, catagory_backward_caption_tokens, catagory_caption_lengths
            )
            
            
            backward_loss = self.loss(
                backward_output_logits[:, :-1].contiguous().view(-1, self.textual.vocab_size),
                catagory_backward_caption_tokens[:, 1:].contiguous().view(-1),
            )
            loss = loss.view(batch_size,-1).sum(dim=1)
            backward_loss = backward_loss.view(batch_size,-1).sum(dim=1)
            
            total_scores = (-loss - backward_loss)/catagory_caption_lengths
            
            
            #print("loss.shape:",loss.shape)
            #print("backward_loss.shape:",backward_loss.shape)
            #print("loss.shape:",loss.shape)
            
            #scores_caption = [torch.sum(x) for x in torch.chunk(loss, batch_size)]
            #scores_noipac = [torch.sum(x) for x in torch.chunk(backward_loss, batch_size)]
            
            #total_scores = [(scores_caption[j]+scores_noipac[j]).item() for j in range(batch_size)]
            
            classification_losses.append(total_scores)
            
            
        #classification_losses = torch.tensor(classification_losses)
        classification_losses = torch.stack(classification_losses).t()

        return classification_losses

   
