import pickle as pkl
import glob
from dataset import SocialNormDataset

from utils.read_files import *
from utils.utils import *
from joblib import Parallel, delayed
import pandas as pd
from constants import *
from sentence_transformers import SentenceTransformer
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--path_to_data", dest="path_to_data", required=True, type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    path_to_data = args.path_to_data
    social_chemistry = pd.read_pickle(path_to_data +'social_chemistry_clean_with_fulltexts.gzip', compression='gzip')

    with open(path_to_data+'social_norms_clean.csv') as file:
        social_comments = pd.read_csv(file)
    
    dataset = SocialNormDataset(social_comments, social_chemistry)
    model = SentenceTransformer('all-MiniLM-L6-v2').cpu()
    
    verdicts_embeddings = dict()
    c = 0
    progress_bar = tqdm(range(len(dataset.verdictToCleanedText)), desc="Creating embeddings")
    for verdict, text in dataset.verdictToCleanedText.items():
        with torch.no_grad():
            embedding = model.encode(process_tweet(text), show_progress_bar=False)
        verdicts_embeddings[verdict] = embedding
        progress_bar.update(1)
        if c == 0:
            print(embedding)
            c += 1
        
    pkl.dump(verdicts_embeddings, open(os.path.join(path_to_data, 'embeddings/verdicts_social_chemistry.pkl'), 'wb'))
        
    
