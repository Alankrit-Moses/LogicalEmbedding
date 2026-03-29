import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class PolarLogicManifold(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_logic_layers=4):
        super().__init__()
        
        # 1. Frozen Semantic Backbone (Layers 0-11)
        self.bert = BertModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # 2. Trainable <C> Token
        self.hidden_size = self.bert.config.hidden_size
        self.c_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        
        # 3. Trainable Logic Head (Layers 12-15)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, 
            nhead=12, 
            dim_feedforward=self.hidden_size * 4,
            activation='gelu',
            batch_first=True
        )
        self.logic_head = nn.TransformerEncoder(encoder_layer, num_layers=num_logic_layers)

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        
        # Phase 1: Frozen Features
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            
        # Phase 2: Token Injection (Layout A: [C] + [CLS] + Words)
        c_tokens = self.c_token.expand(batch_size, -1, -1)
        new_hidden = torch.cat([c_tokens, hidden_states], dim=1)
        
        c_mask = torch.ones((batch_size, 1), device=input_ids.device, dtype=attention_mask.dtype)
        new_mask = torch.cat([c_mask, attention_mask], dim=1)
        
        # PyTorch requires True for ignored padding tokens
        src_key_padding_mask = (new_mask == 0)
        
        # Phase 3: Logical Projection
        logic_outputs = self.logic_head(new_hidden, src_key_padding_mask=src_key_padding_mask)
        
        # Extract the <C> token (Index 0)
        return logic_outputs[:, 0, :]


class AngularMagnitudeLoss(nn.Module):
    def __init__(self, margin=1.0, mag_weight=0.5):
        super().__init__()
        self.margin = margin
        self.mag_weight = mag_weight

    def forward(self, v_p, v_h, labels):
        # 1. Angular Loss
        cos_sim = F.cosine_similarity(v_p, v_h, dim=-1)
        target_cos = torch.where(labels == 0, torch.tensor(1.0, device=v_p.device),
                     torch.where(labels == 1, torch.tensor(0.0, device=v_p.device),
                                              torch.tensor(-1.0, device=v_p.device)))
        
        loss_ang = F.mse_loss(cos_sim, target_cos)

        # 2. Magnitude Loss (Only applied when Premise entails Hypothesis)
        mag_p = torch.norm(v_p, dim=-1)
        mag_h = torch.norm(v_h, dim=-1)
        
        entail_mask = (labels == 0).float()
        mag_diff = mag_p - mag_h
        hinge = F.relu(self.margin - mag_diff)
        
        loss_mag = (hinge * entail_mask).sum() / (entail_mask.sum() + 1e-8)
        
        return loss_ang + (self.mag_weight * loss_mag), loss_ang, loss_mag