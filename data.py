import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import BertTokenizer
from datasets import load_dataset, concatenate_datasets

class NLIDataset(Dataset):
    def __init__(self, split='train'):
        if split == 'train':
            mnli = load_dataset("glue", "mnli", split="train")
            snli = load_dataset("snli", split="train")
        else:
            mnli = load_dataset("glue", "mnli", split="validation_mismatched")
            snli = load_dataset("snli", split="validation")

        mnli = mnli.select_columns(['premise', 'hypothesis', 'label'])
        snli = snli.select_columns(['premise', 'hypothesis', 'label'])

        combined = concatenate_datasets([mnli, snli])
        # Filter out invalid consensus labels (-1)
        self.data = combined.filter(lambda x: x['label'] in [0, 1, 2])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class DynamicPaddingCollator:
    def __init__(self, tokenizer_name='bert-base-uncased', max_length=128):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __call__(self, batch):
        premises = [item['premise'] for item in batch]
        hypotheses = [item['hypothesis'] for item in batch]
        labels = [item['label'] for item in batch]

        p_tokens = self.tokenizer(premises, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        h_tokens = self.tokenizer(hypotheses, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")

        return {
            "p_input_ids": p_tokens["input_ids"],
            "p_attention_mask": p_tokens["attention_mask"],
            "h_input_ids": h_tokens["input_ids"],
            "h_attention_mask": h_tokens["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def create_dataloaders(batch_size=256):
    train_dataset = NLIDataset(split='train')
    val_dataset = NLIDataset(split='val')
    collator = DynamicPaddingCollator()
    
    # Slurm DDP setup requires DistributedSampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=collator, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, collate_fn=collator, num_workers=8, pin_memory=True)
    
    return train_loader, val_loader