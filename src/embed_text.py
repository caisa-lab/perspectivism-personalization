from argparse import ArgumentParser
import pickle as pkl

from transformers import AutoModel, AutoTokenizer
from dataset import SocialNormDataset

from utils.read_files import *
from utils.train_utils import mean_pooling
from utils.utils import *
import pandas as pd
from constants import *
import torch.nn.functional as F
from joblib import Parallel, delayed


parser = ArgumentParser()
parser.add_argument("--path_to_data", dest="path_to_data", required=True, type=str)
parser.add_argument("--text_type", dest="text_type", required=True, type=str)
parser.add_argument("--sbert_model", dest="sbert_model", default='sentence-transformers/all-distilroberta-v1', type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    path_to_data = args.path_to_data
    social_chemistry = pd.read_pickle(path_to_data +'social_chemistry_clean_with_fulltexts.gzip', compression='gzip')

    with open(path_to_data+'social_norms_clean.csv') as file:
        social_comments = pd.read_csv(file)
    
    dataset = SocialNormDataset(social_comments, social_chemistry)
    sbert_model = args.sbert_model
    tokenizer = AutoTokenizer.from_pretrained(sbert_model)
    model = AutoModel.from_pretrained(sbert_model).to(DEVICE)
    
    if args.text_type == 'full_text':
        idToTextDict = dataset.postIdToText
    elif args.text_type == 'title':
        idToTextDict = dataset.postIdToTitle
    elif args.text_type == 'verdicts':
        idToTextDict = dataset.verdictToCleanedText
    elif args.text_type == 'history':
        print("History type")
        authors = set(dataset.authorsToVerdicts.keys())
        authors.remove('Judgement_Bot_AITA')
        dirname = os.path.join(path_to_data, 'amit_filtered_history/')
        print(f'Processing json files from directory {dirname}')
        filenames = sorted(glob.glob(os.path.join(dirname, '*.json')))
        results = Parallel(n_jobs=32)(delayed(extract_authors_vocab_AMIT)(filename, authors) for filename in tqdm(filenames, desc='Reading files'))

        # merge results
        print("Merging results")
        authors_vocab = ListDict()
        for r in results:
            authors_vocab.update_lists(r)
            
        idToTextDict = dict()
        for authors, vocab in authors_vocab.items():
            for v in vocab:
                idToTextDict[v[1]] = v[0]
    else:
        raise Exception("Wrong text type. Text type should be any of [full_text, title, verdicts, history]") 
            
    DEBUG = True
    embeddings_dict = dict()
    progress_bar = tqdm(range(len(idToTextDict)), desc="Creating embeddings")
    
    batch_size = 64
    batch_ids = []
    batch_texts = []

    for idx, text in idToTextDict.items():
        if len(batch_texts) == batch_size:
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                encoded_input = {k: v.to(DEVICE) for k, v in encoded_input.items()}
                model_output = model(**encoded_input)
                # Perform pooling
                text_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
                # Normalize embeddings
                text_embedding = F.normalize(text_embedding, p=2, dim=1)

            for i, id in enumerate(batch_ids):
                embeddings_dict[id] = text_embedding[i].squeeze().cpu().numpy()
            
            batch_ids = []
            batch_texts = []
    
        batch_ids.append(idx)
        batch_texts.append(process_tweet(text))
        
        progress_bar.update(1)

        
    pkl.dump(embeddings_dict, open(os.path.join(path_to_data, f'embeddings/roberta_{args.text_type}.pkl'), 'wb'))
        
    
