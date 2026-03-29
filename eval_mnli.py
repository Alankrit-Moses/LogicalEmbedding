import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import PolarLogicManifold
from data import NLIDataset, DynamicPaddingCollator

def evaluate_in_distribution(checkpoint_path, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}...")

    # 1. Load Model
    model = PolarLogicManifold().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 2. Load the MNLI/SNLI Validation Set
    # Note: data.py maps 'val' to mnli_validation_mismatched and snli_validation
    val_dataset = NLIDataset(split='val')
    collator = DynamicPaddingCollator()
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collator, num_workers=4)

    correct = 0
    total = 0

    print("Running evaluation on standard validation sets...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            p_ids = batch["p_input_ids"].to(device)
            p_mask = batch["p_attention_mask"].to(device)
            h_ids = batch["h_input_ids"].to(device)
            h_mask = batch["h_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Get Manifold Vectors
            v_p = model(p_ids, p_mask)
            v_h = model(h_ids, h_mask)

            # Calculate Cosine Similarity
            cos_sim = F.cosine_similarity(v_p, v_h, dim=-1)

            # Map Cosine to the Closest Pole Class
            # Class 0 (Entailment) target = 1.0
            # Class 1 (Neutral) target = 0.0
            # Class 2 (Contradiction) target = -1.0
            
            dist_0 = (cos_sim - 1.0)**2
            dist_1 = (cos_sim - 0.0)**2
            dist_2 = (cos_sim - (-1.0))**2

            distances = torch.stack([dist_0, dist_1, dist_2], dim=1)
            predictions = torch.argmin(distances, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = (correct / total) * 100
    print(f"\n--- Validation Results ---")
    print(f"Total Evaluated: {total}")
    print(f"In-Distribution Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    # Point this to your best Smooth L1 checkpoint
    evaluate_in_distribution("checkpoints/plm_epoch_5.pt")