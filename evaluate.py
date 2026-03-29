import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from model import PolarLogicManifold

def evaluate_hans(checkpoint_path):
    # Auto-detect if we are on a Slurm compute node or a login node
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading PLM checkpoint onto {device}...")
    
    # 1. Load Model and Weights
    model = PolarLogicManifold().to(device)
    
    try:
        # Since we saved using model.module.state_dict() in DDP, this should load cleanly
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Weights loaded successfully.\n")
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return
        
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 2. The HANS Heuristic Trap
    # 100% Word overlap, but fundamentally different syntactic logic.
    premise = "The doctor visited the lawyer."
    hypothesis = "The lawyer visited the doctor."
    
    p_tokens = tokenizer(premise, return_tensors="pt").to(device)
    h_tokens = tokenizer(hypothesis, return_tensors="pt").to(device)
    
    # 3. Extract the Manifold Vectors
    with torch.no_grad():
        v_p = model(p_tokens['input_ids'], p_tokens['attention_mask'])
        v_h = model(h_tokens['input_ids'], h_tokens['attention_mask'])
        
        # Calculate our two target metrics
        cos_sim = F.cosine_similarity(v_p, v_h, dim=-1).item()
        mag_p = torch.norm(v_p, dim=-1).item()
        mag_h = torch.norm(v_h, dim=-1).item()
        
    # 4. Print the Diagnostic Report
    print("-" * 50)
    print(f"Premise:    {premise}")
    print(f"Hypothesis: {hypothesis}")
    print("-" * 50)
    print(f"Angular Output (Cosine): {cos_sim:.4f}")
    print(f"Premise Magnitude:       {mag_p:.4f}")
    print(f"Hypothesis Magnitude:    {mag_h:.4f}")
    print(f"Magnitude Difference:    {mag_p - mag_h:.4f}")
    print("-" * 50)
    
    # 5. Interpret the Results based on the Manifold
    if cos_sim > 0.7:
        print("Verdict: FAILED. The model fell for the word-overlap trap (Entailment).")
    elif cos_sim < -0.5:
        print("Verdict: SUCCESS. Manifold logic held the vectors apart (Contradiction).")
    else:
        print("Verdict: COMPRESSED. The model pushed them to the Neutral zone.")

if __name__ == "__main__":
    evaluate_hans("checkpoints/plm_epoch_5.pt")