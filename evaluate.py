import torch
import numpy as np
from tqdm.auto import tqdm
from model import PLMModel
from data import get_dataloaders
import argparse

def evaluate_angular_spread(model, dataloader, device):
    model.eval()
    all_angles = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            input_ids_a = batch['input_ids_a'].to(device)
            attention_mask_a = batch['attention_mask_a'].to(device)
            input_ids_b = batch['input_ids_b'].to(device)
            attention_mask_b = batch['attention_mask_b'].to(device)
            labels = batch['labels'].to(device)
            
            emb_a = model(input_ids_a, attention_mask_a)
            emb_b = model(input_ids_b, attention_mask_b)
            
            # Calculate cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(emb_a, emb_b)
            # Clip for safety before acos
            cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
            
            # Convert to degrees
            angles = torch.acos(cos_sim) * (180.0 / np.pi)
            
            all_angles.extend(angles.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    all_angles = np.array(all_angles)
    all_labels = np.array(all_labels)
    
    # 0: Entailment, 1: Neutral, 2: Contradiction
    results = {}
    for label_idx, label_name in [(0, "Entailment"), (1, "Neutral"), (2, "Contradiction")]:
        mask = (all_labels == label_idx)
        if mask.any():
            avg_angle = np.mean(all_angles[mask])
            std_angle = np.std(all_angles[mask])
            results[label_name] = {"avg": avg_angle, "std": std_angle}
            
    return results, all_angles, all_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt file")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = PLMModel(model_name=args.model_name)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    
    # Load data
    _, val_dataloader, _ = get_dataloaders(batch_size=args.batch_size, model_name=args.model_name)
    
    results, angles, labels = evaluate_angular_spread(model, val_dataloader, device)
    
    print("\n--- Polar Logic Manifold (PLM) Evaluation ---")
    for label_name, metrics in results.items():
        print(f"{label_name:15}: Avg Angle = {metrics['avg']:.2f} deg, Std Dev = {metrics['std']:.2f}")
        
    # Categorize by angular spread
    print("\n--- Angular Spread Analysis ---")
    # [0, 45) deg: Leaning Entailment
    # [45, 90) deg: Leaning Neutral
    # (90, 135] deg: Leaning Neutral
    # (135, 180] deg: Leaning Contradiction
    
    bins = [0, 45, 90, 135, 180]
    bin_names = ["Leaning Entailment", "Leaning Neutral (Low)", "Leaning Neutral (High)", "Leaning Contradiction"]
    
    for i in range(len(bins)-1):
        mask = (angles >= bins[i]) & (angles < bins[i+1])
        count = np.sum(mask)
        if count > 0:
            # Distribution of true labels in this bin
            bin_labels = labels[mask]
            label_dist = {0: 0, 1: 0, 2: 0}
            for l in bin_labels:
                label_dist[l] += 1
            
            print(f"{bin_names[i]:25}: {count} samples")
            print(f"  True Label Dist: Ent={label_dist[0]}, Neut={label_dist[1]}, Contr={label_dist[2]}")

if __name__ == "__main__":
    main()
