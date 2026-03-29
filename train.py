import os
import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from model import PolarLogicManifold, AngularMagnitudeLoss
from data import create_dataloaders

def setup_slurm_ddp():
    """Initializes DDP using Slurm environment variables."""
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def train(epochs=5, batch_size=256, accumulation_steps=2):
    local_rank = setup_slurm_ddp()
    is_master = (local_rank == 0)
    
    model = PolarLogicManifold().to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    criterion = AngularMagnitudeLoss(margin=1.0).to(local_rank)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

    train_loader, val_loader = create_dataloaders(batch_size=batch_size)

    for epoch in range(epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch)
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            p_ids = batch["p_input_ids"].to(local_rank)
            p_mask = batch["p_attention_mask"].to(local_rank)
            h_ids = batch["h_input_ids"].to(local_rank)
            h_mask = batch["h_attention_mask"].to(local_rank)
            labels = batch["labels"].to(local_rank)

            my_context = model.no_sync() if (step + 1) % accumulation_steps != 0 else torch.enable_grad()
            
            with my_context:
                v_p = model(p_ids, p_mask)
                v_h = model(h_ids, h_mask)
                
                loss, loss_ang, loss_mag = criterion(v_p, v_h, labels)
                loss = loss / accumulation_steps
                loss.backward()

            if (step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            if is_master and step % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Step [{step}/{len(train_loader)}] "
                      f"Total Loss: {loss.item() * accumulation_steps:.4f} "
                      f"(Angle: {loss_ang.item():.4f}, Mag: {loss_mag.item():.4f})")

        if is_master:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.module.state_dict(), f"checkpoints/plm_epoch_{epoch+1}.pt")
            print(f"Saved Checkpoint for Epoch {epoch+1}")

    cleanup_ddp()

if __name__ == "__main__":
    train(epochs=5, batch_size=256, accumulation_steps=2)