import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from model import PolarLogicManifold

def evaluate_hans(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model and weights
    model = PolarLogicManifold().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # A classic HANS heuristic trap (100% word overlap, but logical contradiction/neutral)
    premise = "The doctor visited the lawyer."
    hypothesis = "The lawyer visited the doctor."
    
    p_tokens = tokenizer(premise, return_tensors="pt").to(device)
    h_tokens = tokenizer(hypothesis, return_tensors="pt").to(device)
    
    with torch.no_grad():
        v_p = model(p_tokens['input_ids'], p_tokens['attention_mask'])
        v_h = model(h_tokens['input_ids'], h_tokens['attention_mask'])
        
        cos_sim = F.cosine_similarity(v_p, v_h, dim=-1).item()
        
    print(f"Premise: {premise}")
    print(f"Hypothesis: {hypothesis}")
    print(f"Cosine Similarity: {cos_sim:.4f}")
    
    if cos_sim > 0.7:
        print("Verdict: The model fell for the overlap trap (Entailment).")
    elif cos_sim < -0.5:
        print("Verdict: Manifold logic holds! (Contradiction).")
    else:
        print("Verdict: Manifold logic holds! (Neutral separation).")

if __name__ == "__main__":
    # Point this to your saved weights after training
    evaluate_hans("checkpoints/plm_epoch_1.pt")