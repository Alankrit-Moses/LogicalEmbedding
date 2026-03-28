import torch
import torch.nn as nn
import torch.nn.functional as F

class PLMLoss(nn.Module):
    def __init__(self, k=0.1, m_min=1.0, alpha=1.0, beta=1.0, gamma=0.1):
        super(PLMLoss, self).__init__()
        self.k = k  # Magnitude margin for specificity
        self.m_min = m_min  # Minimum norm
        self.alpha = alpha  # Angular loss weight
        self.beta = beta  # Magnitude loss weight
        self.gamma = gamma  # Min norm loss weight

    def forward(self, emb_a, emb_b, labels):
        """
        emb_a, emb_b: [batch_size, hidden_size]
        labels: [batch_size] - 0: Entailment, 1: Neutral, 2: Contradiction
        """
        # --- Angular Polarity (Theta) ---
        cos_sim = F.cosine_similarity(emb_a, emb_b)
        
        # Targets: Entailment (0) -> 1.0, Neutral (1) -> 0.0, Contradiction (2) -> -1.0
        # We can use MSE or a custom loss for this
        target_cos = torch.zeros_like(cos_sim)
        target_cos[labels == 0] = 1.0
        target_cos[labels == 1] = 0.0
        target_cos[labels == 2] = -1.0
        
        # For Neutral, we want cos_sim to be close to 0.0
        # For others, we want them to be close to 1.0 or -1.0
        loss_angular = F.mse_loss(cos_sim, target_cos)
        
        # --- Magnitude Logic (||V||) ---
        norm_a = torch.norm(emb_a, p=2, dim=1)
        norm_b = torch.norm(emb_b, p=2, dim=1)
        
        # 1. Specificity (A entails B -> ||A|| > ||B||)
        # NLI usually has Premise -> Hypothesis, so A is more specific
        # L_spec = max(0, k - (||A|| - ||B||))
        loss_spec = torch.clamp(self.k - (norm_a - norm_b), min=0.0)
        # Only apply specificity loss for Entailment
        loss_spec = loss_spec[labels == 0].mean() if (labels == 0).any() else torch.tensor(0.0, device=emb_a.device)
        
        # 2. Minimum Norm
        loss_min_norm = torch.clamp(self.m_min - norm_a, min=0.0).mean() + \
                        torch.clamp(self.m_min - norm_b, min=0.0).mean()
        
        # --- Total Loss ---
        total_loss = self.alpha * loss_angular + self.beta * loss_spec + self.gamma * loss_min_norm
        
        return total_loss, {
            "loss_angular": loss_angular.item(),
            "loss_spec": loss_spec.item(),
            "loss_min_norm": loss_min_norm.item()
        }
