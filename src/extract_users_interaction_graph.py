import pandas as pd
from utils.clusters_utils import *
from dataset import SocialNormDataset
import glob
from joblib import Parallel, delayed
import os
from utils.read_files import *
import json
from argparse import ArgumentParser

"""At the moment all paths are hard coded. Needs input/output arguments for filepaths.  
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
    
    
    dirname = os.path.join(path_to_data, 'amit_filtered_history/')
    authors = set(dataset.authorsToVerdicts.keys())
    print(len(authors))
    filenames = sorted(glob.glob(os.path.join(dirname, '*.json')))
    results = Parallel(n_jobs=32)(delayed(extract_authors_vocab_AMIT)(filename, authors) for filename in tqdm(filenames, desc='Reading files'))
    # merge results
    print("Merging results")
    authors_vocab = ListDict()
    for r in results:
        authors_vocab.update_lists(r)

    print(len(authors_vocab))
    
    authorsToSituations = SetDict()
    for author, vocab in authors_vocab.items():
        for v in vocab:
            authorsToSituations.add(author, v[-1])
            
    
    authorToAuthors = ListDict()

    for author, situations in tqdm(authorsToSituations.items()):
        for target_author, target_sit in authorsToSituations.items():
            if author != target_author:
                common = len(situations.intersection(target_sit))
                if common > 0:
                    authorToAuthors.append(author, (target_author, common))
                    
    json.dump(authorToAuthors, open(os.path.join(path_to_data, 'authors_interactions_in_history.json'), 'w'))