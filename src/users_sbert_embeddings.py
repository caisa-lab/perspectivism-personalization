import pandas as pd
import glob
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle as pkl

from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
from argparse import ArgumentParser
from dataset import SocialNormDataset
from models import MLPAttribution

from utils.read_files import *
from utils.train_utils import mean_pooling
from utils.utils import *
from constants import *

parser = ArgumentParser()
parser.add_argument("--path_to_data", dest="path_to_data", required=True, type=str)

parser.add_argument("--age_dir", dest="age_dir", default='../data/demographic/age_list.csv', type=str)
parser.add_argument("--gender_dir", dest="gender_dir", default='../data/demographic/gender_list.csv', type=str)
parser.add_argument("--bert_model", dest="bert_model", default='sentence-transformers/all-distilroberta-v1', type=str)
parser.add_argument("--age_gender_authors", dest="age_gender_authors", type=str2bool, default=False)
parser.add_argument("--dirname", dest="dirname", type=str, required=True)
parser.add_argument("--output_dir", dest="output_dir", type=str, required=True)

if __name__ == '__main__':
    args = parser.parse_args()
    path_to_data = args.path_to_data
    age_gender_authors = False
    
    if age_gender_authors:
        age_df = pd.read_csv(args.age_dir, sep='\t', names=['author', 'subreddit', 'age'])
        gender_df = pd.read_csv(args.gender_dir, sep='\t', names=['author', 'gender'])
        age_authors = set(age_df.author)
        gender_authors = set(gender_df.author)
        authors = gender_authors.intersection(age_authors)
    else:
        social_chemistry = pd.read_pickle(path_to_data +'social_chemistry_clean_with_fulltexts.gzip', compression='gzip')

        with open(path_to_data+'social_norms_clean.csv') as file:
            social_comments = pd.read_csv(file)
            
        dataset = SocialNormDataset(social_comments, social_chemistry)
        authors = set(dataset.authorsToVerdicts.keys())
    print(DEVICE)
    print(len(authors))
    
    if 'amit' in args.dirname:
        print(f'Processing json files from directory {args.dirname}')
        filenames = sorted(glob.glob(os.path.join(args.dirname, '*.json')))
        results = Parallel(n_jobs=32)(delayed(extract_authors_vocab_AMIT)(filename, authors) for filename in tqdm(filenames, desc='Reading files'))
    else:
        print(f'Processing text files from directory {args.dirname}')
        filenames = sorted(glob.glob(os.path.join(args.dirname, '*')))
        results = Parallel(n_jobs=32)(delayed(extract_authors_vocab_notAMIT)(filename, authors) for filename in tqdm(filenames))

    # merge results
    print("Merging results")
    authors_vocab = ListDict()
    for r in results:
        authors_vocab.update_lists(r)

    print(len(authors_vocab))
    
    print("Using {} model".format(args.bert_model))
    
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    model = AutoModel.from_pretrained(args.bert_model).to(DEVICE)
    
    user_embeddings = {}
    
    def extract_batches(seq, batch_size=32):
        n = len(seq) // batch_size
        batches  = []

        for i in range(n):
            batches.append(seq[i * batch_size:(i+1) * batch_size])
        if len(seq) % batch_size != 0:
            batches.append(seq[n * batch_size:])
        return batches
        
    
    DEBUG = True
    for author, texts in tqdm(authors_vocab.items(), desc="Embedding authors"):
        processed_texts = [process_tweet(text[0]) for text in texts]
        # Tokenize sentences
        batches_text = extract_batches(processed_texts, 64)
        embeddings = []
        encoded_inputs = [tokenizer(processed_texts, padding=True, truncation=True, return_tensors='pt') for processed_texts in batches_text]

        for encoded_input in encoded_inputs:
            with torch.no_grad():
                # Compute token embeddings
                encoded_input = {k: v.to(DEVICE) for k, v in encoded_input.items()}
                model_output = model(**encoded_input)
                # Perform pooling
                sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

                # Normalize embeddings
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            
                
                average = sentence_embeddings.cpu().mean(axis=0)
                embeddings.append(average.unsqueeze(0))

        if len(embeddings) > 1:
            embedding = torch.cat(embeddings)
            user_embeddings[author] = embedding.mean(axis=0).numpy()
        else:
            user_embeddings[author] = embeddings[0].squeeze().numpy()
        
        if DEBUG:
            print(user_embeddings[author], user_embeddings[author].shape)
            DEBUG = False
        
    
    print("Saving embeddings")
    pkl.dump(user_embeddings, open(args.output_dir, 'wb'))
    
    