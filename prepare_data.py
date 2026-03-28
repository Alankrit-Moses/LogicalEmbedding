from datasets import load_dataset, concatenate_datasets
from transformers import BertTokenizer
import os

def prepare():
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    print("Loading datasets...")
    snli = load_dataset("snli")
    mnli = load_dataset("multi_nli")
    
    # Filter and Combine
    def filter_label(example): return example['label'] != -1
    train_data = concatenate_datasets([snli['train'].filter(filter_label), mnli['train'].filter(filter_label)])
    val_data = concatenate_datasets([snli['validation'].filter(filter_label), mnli['validation_matched'].filter(filter_label)])

    def tokenize_function(examples):
        res_a = tokenizer(examples['premise'], padding='max_length', truncation=True, max_length=128)
        res_b = tokenizer(examples['hypothesis'], padding='max_length', truncation=True, max_length=128)
        return {
            'input_ids_a': res_a['input_ids'], 'attention_mask_a': res_a['attention_mask'],
            'input_ids_b': res_b['input_ids'], 'attention_mask_b': res_b['attention_mask'],
            'labels': examples['label']
        }

    print("Tokenizing (this might take a few minutes)...")
    train_data = train_data.map(tokenize_function, batched=True, num_proc=8)
    val_data = val_data.map(tokenize_function, batched=True, num_proc=8)

    print("Saving to disk...")
    os.makedirs("data/processed", exist_ok=True)
    train_data.save_to_disk("data/processed/train")
    val_data.save_to_disk("data/processed/val")
    
    print("\nPre-tokenized data saved to data/processed/. Your training job will now start instantly!")

if __name__ == "__main__":
    prepare()
