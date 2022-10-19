import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm 
import torch
import numpy as np
from dataset import SocialNormDataset
from constants import *
from datasets import DatasetDict, Dataset, Features, Value
from transformers import AutoTokenizer, DataCollatorWithPadding, BertTokenizerFast
from torch.utils.data import DataLoader
from models import SentBertClassifier
import pickle as pkl
from argparse import ArgumentParser


""" This is a script to extract verdicts plus situations embeddings for training classifiers.
"""
parser = ArgumentParser()
parser.add_argument("--path_to_data", dest="path_to_data", required=True, type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    path_to_data = args.path_to_data
    
    social_chemistry = pd.read_pickle(path_to_data +'social_chemistry_clean_with_fulltexts.gzip', compression='gzip')

    with open(path_to_data+'social_norms_clean.csv') as file:
        social_comments = pd.read_csv(file)
        

    dataset = SocialNormDataset(social_comments, social_chemistry)
    raw_dataset = {'full': {'index': [], 'text': [], 'label': []}}
    flag = False
    
    if flag:
        verdict_ids = list(dataset.verdictToLabel.keys())
        labels = list(dataset.verdictToLabel.values())
        for i, verdict in enumerate(verdict_ids):
            situation_text = dataset.postIdToText[dataset.verdictToParent[verdict]]
            if situation_text != '' and situation_text is not None: 
                raw_dataset['full']['index'].append(dataset.verdictToId[verdict])
                raw_dataset['full']['text'].append(situation_text + ' [SEP] ' + dataset.verdictToCleanedText[verdict])
                raw_dataset['full']['label'].append(labels[i])
                assert labels[i] == dataset.verdictToLabel[verdict] 
    else:
        print("Embeddings full texts of situations")
        post_ids = list(dataset.postIdToText.keys())
        
        for i, post in enumerate(post_ids):
            situation_text = dataset.clean_single_text(dataset.postIdToText[post])
            if situation_text != '' and situation_text is not None:
                raw_dataset['full']['index'].append(dataset.postIdToId[post])
                raw_dataset['full']['text'].append((situation_text))
                raw_dataset['full']['label'].append(0)
                
            
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    ds = DatasetDict()

    for split, data in raw_dataset.items():
        ds[split] = Dataset.from_dict(mapping=data, features=Features({'label': Value(dtype='int64'), 
                                                                        'text': Value(dtype='string'), 'index': Value(dtype='int64')}))
        
    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True)

    tokenized_dataset = ds.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    batch_size = 64

    dataloader = DataLoader(
        tokenized_dataset["full"], batch_size=batch_size, collate_fn=data_collator
    )
    
    sbert_model = 'sentence-transformers/all-MiniLM-L6-v2' 
    sbert_dim = 384

    model = SentBertClassifier(users_layer=False, sbert_model=sbert_model, sbert_dim=sbert_dim)
    #checkpoint_dir = '../results/best_models/2022-05-15_18:49:14:232687_best_model_sampled.pt' # situations stratified
    checkpoint_dir = '../results/best_models/2022-05-16_10:22:18:785822_best_model_sampled.pt' # fulltexts stratified
    print("*** Loading from {} ***".format(checkpoint_dir))
    model.load_state_dict(torch.load(checkpoint_dir))
    model.to(DEVICE)
    print("Loaded model")
    
    
    model.eval()
    
    verdictToEmbedding = {}

    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        verdicts_index = batch.pop("index")
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        labels = batch.pop("labels")
        with torch.no_grad():
            bert_output = model.model(**batch)
            pooled_output = model.mean_pooling(bert_output, batch['attention_mask'])
        
        for i, idx in enumerate(verdicts_index):
            # verdict = dataset.idToVerdict[idx]
            # verdictToEmbedding[verdict] = pooled_output[i].squeeze().cpu()
            post = dataset.idTopostId[idx]
            verdictToEmbedding[post] = pooled_output[i].squeeze().cpu()
           
           
    pkl.dump(verdictToEmbedding, open(f'{path_to_data}/embeddings/posts_fullTextSBert.pkl', 'wb'))