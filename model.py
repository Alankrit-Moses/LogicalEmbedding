import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class PLMModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_extra_layers=4, hidden_size=768):
        super(PLMModel, self).__init__()
        
        # 1. Frozen Backbone (Layers 1-12)
        self.backbone = BertModel.from_pretrained(model_name)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 2. Constraint <C> Token
        # Initialized as a trainable parameter
        self.c_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        # 3. Trainable Head (Layers 13-16)
        # Using standard Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=12,  # Matching BERT-base configuration
            dim_feedforward=3072,  # Matching BERT-base configuration
            batch_first=True,
            activation='gelu'
        )
        self.trainable_head = nn.TransformerEncoder(encoder_layer, num_layers=num_extra_layers)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Pass through the frozen backbone
        # We need the full hidden states or at least the last layer's output
        with torch.no_grad():
            outputs = self.backbone(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
            
        # Inject <C> token
        batch_size = sequence_output.size(0)
        c_tokens = self.c_token.expand(batch_size, -1, -1)  # [batch_size, 1, hidden_size]
        
        # Prepend <C> token to the sequence
        # We also need to update the attention mask for the extra token
        extended_sequence = torch.cat((c_tokens, sequence_output), dim=1)  # [batch_size, seq_len + 1, hidden_size]
        
        if attention_mask is not None:
            # Prepend a '1' to the attention mask for the <C> token
            c_mask = torch.ones((batch_size, 1), device=attention_mask.device)
            extended_attention_mask = torch.cat((c_mask, attention_mask), dim=1)
        else:
            extended_attention_mask = None
            
        # Pass through the trainable head
        # TransformerEncoder expects src_key_padding_mask as (batch, seq_len) bool where True is padding
        padding_mask = (extended_attention_mask == 0) if extended_attention_mask is not None else None
        
        head_output = self.trainable_head(extended_sequence, src_key_padding_mask=padding_mask)
        
        # Extract the <C> token output as the embedding
        # It was prepended, so it's at index 0
        embeddings = head_output[:, 0, :]
        
        return embeddings
