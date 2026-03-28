from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import torch

def get_dataloaders_online(batch_size=32, model_name='bert-base-uncased'):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    snli = load_dataset("snli")
    mnli = load_dataset("multi_nli")
    
    def filter_label(example): return example['label'] != -1
    snli = snli.filter(filter_label)
    mnli = mnli.filter(filter_label)
    
    train_data = concatenate_datasets([snli['train'], mnli['train']])
    val_data = concatenate_datasets([snli['validation'], mnli['validation_matched']])
    
    def tokenize_function(examples):
        res_a = tokenizer(examples['premise'], padding='max_length', truncation=True, max_length=128)
        res_b = tokenizer(examples['hypothesis'], padding='max_length', truncation=True, max_length=128)
        return {
            'input_ids_a': res_a['input_ids'], 'attention_mask_a': res_a['attention_mask'],
            'input_ids_b': res_b['input_ids'], 'attention_mask_b': res_b['attention_mask'],
            'labels': examples['label']
        }
        
    train_data = train_data.map(tokenize_function, batched=True)
    val_data = val_data.map(tokenize_function, batched=True)
    
    train_data.set_format(type='torch')
    val_data.set_format(type='torch')
    
    return DataLoader(train_data, batch_size=batch_size, shuffle=True), \
           DataLoader(val_data, batch_size=batch_size), \
           tokenizer
