import torch
import torch.optim as optim
from accelerate import Accelerator
from tqdm.auto import tqdm
import argparse
import os

from model import PLMModel
from loss import PLMLoss
from data import get_dataloaders

def main():
    parser = argparse.ArgumentParser(description="Train PLM Model")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Base model name")
    parser.add_argument("--k", type=float, default=0.1, help="Magnitude margin for specificity")
    parser.add_argument("--m_min", type=float, default=1.0, help="Minimum norm")
    parser.add_argument("--alpha", type=float, default=1.0, help="Angular loss weight")
    parser.add_argument("--beta", type=float, default=1.0, help="Magnitude loss weight")
    parser.add_argument("--gamma", type=float, default=0.1, help="Min norm loss weight")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    
    args = parser.parse_args()
    
    # Initialize Accelerator
    # We disable tensorboard logging temporarily to avoid TensorFlow/cuBLAS conflicts seen in logs
    accelerator = Accelerator()
    
    # Setup Device explicitly
    device = accelerator.device
    torch.cuda.set_device(device)
    
    if accelerator.is_local_main_process:
        print(f"Running on {accelerator.num_processes} GPUs")
        print(f"Device: {device}")

    # 1. Model
    # Move model to device before preparation to be safe
    model = PLMModel(model_name=args.model_name).to(device)
    
    # 2. Loss and Optimizer
    criterion = PLMLoss(k=args.k, m_min=args.m_min, alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    # 3. Data
    train_dataloader, val_dataloader, tokenizer = get_dataloaders(batch_size=args.batch_size, model_name=args.model_name)
    
    # 4. Prepare everything with Accelerator
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )
    
    # 5. Training Loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}", disable=not accelerator.is_local_main_process)
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Forward pass for premise (A) and hypothesis (B)
            emb_a = model(batch['input_ids_a'], batch['attention_mask_a'])
            emb_b = model(batch['input_ids_b'], batch['attention_mask_b'])
            
            # Calculate loss
            loss, metrics = criterion(emb_a, emb_b, batch['labels'])
            
            # Backward pass
            accelerator.backward(loss)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Log progress
            progress_bar.set_postfix({"loss": loss.item()})
            accelerator.log({
                "train_loss": loss.item(),
                "loss_angular": metrics["loss_angular"],
                "loss_spec": metrics["loss_spec"],
                "loss_min_norm": metrics["loss_min_norm"]
            })
            
        # 6. Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                emb_a = model(batch['input_ids_a'], batch['attention_mask_a'])
                emb_b = model(batch['input_ids_b'], batch['attention_mask_b'])
                loss, _ = criterion(emb_a, emb_b, batch['labels'])
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_dataloader)
        accelerator.log({"val_loss": avg_val_loss})
        
        if accelerator.is_local_main_process:
            print(f"Epoch {epoch}: Average Val Loss: {avg_val_loss}")
            
            # Save Checkpoint
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), f"{args.output_dir}/plm_model_epoch_{epoch}.pt")
            
    accelerator.end_training()

if __name__ == "__main__":
    main()
