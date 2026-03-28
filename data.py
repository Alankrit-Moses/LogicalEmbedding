from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import os

def get_dataloaders(batch_size=32, model_name='bert-base-uncased'):
    # Check if pre-tokenized data exists
    if os.path.exists("data/processed/train") and os.path.exists("data/processed/val"):
        print("Loading pre-tokenized datasets from disk...")
        train_data = load_from_disk("data/processed/train")
        val_data = load_from_disk("data/processed/val")
    else:
        # Fallback to standard loading if prep script hasn't been run
        # This will be slower as it tokenizes on-the-fly for each process
        from data_legacy import get_dataloaders_online
        return get_dataloaders_online(batch_size, model_name)
    
    train_data.set_format(type='torch')
    val_data.set_format(type='torch')
    
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size)
    
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    return train_dataloader, val_dataloader, tokenizer
