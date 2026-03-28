from datasets import load_dataset
from transformers import BertTokenizer, BertModel

def prepare():
    print("Downloading SNLI dataset...")
    load_dataset("snli")
    
    print("Downloading MultiNLI dataset...")
    load_dataset("multi_nli")
    
    print("Downloading BERT model and tokenizer...")
    model_name = "bert-base-uncased"
    BertTokenizer.from_pretrained(model_name)
    BertModel.from_pretrained(model_name)
    
    print("\nAll data and models have been successfully cached!")

if __name__ == "__main__":
    prepare()
