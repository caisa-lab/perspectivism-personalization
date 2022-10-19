from argparse import ArgumentParser
import pickle as pkl

from transformers import AutoModel, AutoTokenizer
from dataset import SocialNormDataset
from joblib import Parallel, delayed

from utils.read_files import *
from utils.train_utils import mean_pooling
from utils.utils import *
import pandas as pd
from constants import *
import torch.nn.functional as F
from utils.train_utils import *
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from models import MLPAttribution


# python users_attribution_embeddings.py --checkpoint_dir='verdicts_linear_layers_2022-06-16_12:12:57:866975.pt' --embedding_type='prediction'
parser = ArgumentParser()
parser.add_argument("--path_to_data", dest="path_to_data", required=True, type=str)
parser.add_argument("--checkpoint_dir", dest="checkpoint_dir", required=True, type=str)
parser.add_argument("--embedding_type", dest="embedding_type", required=True, type=str) # ['distribution', 'prediction']

if __name__ == '__main__':
    args = parser.parse_args()
    path_to_data = args.path_to_data
    embedding_type = args.embedding_type
    checkpoint_dir = args.checkpoint_dir

    if checkpoint_dir == '':
        raise Exception("No checkpoint dir")
    
    verdict_embedder = pkl.load(open('../data/embeddings/emnlp/roberta_history.pkl', 'rb'))
    # Reading data
    social_chemistry = pd.read_pickle(path_to_data +'social_chemistry_clean_with_fulltexts.gzip', compression='gzip')
    with open(path_to_data+'social_norms_clean.csv') as file:
        social_comments = pd.read_csv(file)
        
    dataset = SocialNormDataset(social_comments, social_chemistry)
    authors = set(dataset.authorsToVerdicts.keys())
    
    dirname = os.path.join(path_to_data, 'amit_filtered_history/')
    
    filenames = sorted(glob.glob(os.path.join(dirname, '*.json')))
    results = Parallel(n_jobs=32)(delayed(extract_authors_vocab_AMIT)(filename, authors) for filename in tqdm(filenames, desc='Reading files'))

    # merge results
    print("Merging results")
    authors_vocab = ListDict()
    for r in results:
        authors_vocab.update_lists(r)

    # Loading the model
    checkpoint = torch.load(os.path.join(path_to_data, f'attribution_models/{checkpoint_dir}'))
    split_type = checkpoint['split']
    output_path = os.path.join(path_to_data, f'embeddings/emnlp/attribution/{split_type}_mlp_{embedding_type}.pkl')
    
    model = MLPAttribution(768, 384, checkpoint['num_output'])
    model.load_state_dict(checkpoint['model'])
    model.to(DEVICE)
    
    user_embeddings = {}
    
    def extract_batches(seq, batch_size=128):
        n = len(seq) // batch_size
        batches  = []

        for i in range(n):
            batches.append(seq[i * batch_size:(i+1) * batch_size])
        if len(seq) % batch_size != 0:
            batches.append(seq[n * batch_size:])
        return batches
        
    
    DEBUG = False
    for author, texts in tqdm(authors_vocab.items(), desc="Embedding authors"):
        verdicts = [text[1] for text in texts]
        # Tokenize sentences
        batches_verdicts = extract_batches(verdicts, 256)
        batch_embeddings = [torch.tensor([verdict_embedder[v] for v in batch if v in verdict_embedder]) for batch in batches_verdicts]
        size = 0
        if embedding_type == 'distribution':
            embeddings = []
        else:
            embeddings = torch.zeros(checkpoint['num_output'])
        
        for batch in batch_embeddings:
            size += batch.size()[0]                 
            with torch.no_grad():
                output = model(batch.to(DEVICE))
                if embedding_type == 'distribution':
                    output = F.normalize(output, p=2, dim=1)
                    embedding = output.cpu().mean(axis=0).unsqueeze(0)
                    embeddings.append(embedding)
                elif embedding_type == 'prediction':
                    predictions = torch.argmax(output, dim=-1)
                    for i in predictions:
                        embeddings[i] += 1
                else:
                    raise Exception("Wrong embedding type")
        
        if embedding_type == 'distribution':     
            if len(embeddings) > 1:
                embeddings = torch.cat(embeddings)
                user_embeddings[author] = embeddings.mean(axis=0).numpy()
            else:
                user_embeddings[author] = embeddings[0].squeeze().numpy()
        elif embedding_type == 'prediction':
            user_embeddings[author] = embeddings / size
        else:
            raise Exception("Wrong embedding type")

        if DEBUG:
            print(batch.size())
            print(user_embeddings[author], user_embeddings[author].shape)
            DEBUG = False
            
    print("Saving embeddings")
    pkl.dump(user_embeddings, open(output_path, 'wb'))
    
    