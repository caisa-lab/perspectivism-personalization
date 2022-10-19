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
import logging

TIMESTAMP = get_current_timestamp()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"../logs/{TIMESTAMP}_attribution.log"),
        logging.StreamHandler()
    ]
)

def evaluate_mlp(model, dataloader):
    accuracy_metric = load_metric("accuracy")
    f1_macro = load_metric("f1")
    f1_weighted = load_metric("f1")
   
    model.eval()     
    for batch in dataloader:
        with torch.no_grad():
            input = batch[0]
            labels = batch[1].to(DEVICE)
            embedding = embedding_layer(input).to(DEVICE)
            logits = model(embedding)

        predictions = torch.argmax(logits, dim=-1)
        accuracy_metric.add_batch(predictions=predictions, references=labels)
        f1_macro.add_batch(predictions=predictions, references=labels)
        f1_weighted.add_batch(predictions=predictions, references=labels)

    return {'accuracy': accuracy_metric.compute()['accuracy'], 'f1_score_macro': f1_macro.compute(average="macro")['f1'], 
        'f1_score_weighted': f1_weighted.compute(average="weighted")['f1']}


parser = ArgumentParser()
parser.add_argument("--split_type", dest="split_type", required=True, type=str) # ['author', 'sit', 'verdicts']
parser.add_argument("--path_to_data", dest="path_to_data", required=True, type=str)

if __name__ == '__main__':
    args = parser.parse_args()

    path_to_data = args.path_to_data
    social_chemistry = pd.read_pickle(path_to_data +'social_chemistry_clean_with_fulltexts.gzip', compression='gzip')
    split_type = args.split_type
    
    with open(path_to_data+'social_norms_clean.csv') as file:
        social_comments = pd.read_csv(file)
        
    dataset = SocialNormDataset(social_comments, social_chemistry)
    authors = set(dataset.authorsToVerdicts.keys())
    if 'Judgement_Bot_AITA' in authors:
        authors.remove('Judgement_Bot_AITA')
    
    dirname = os.path.join(path_to_data, 'amit_filtered_history/')
    logging.info(f'Processing json files from directory {dirname}')
    filenames = sorted(glob.glob(os.path.join(dirname, '*.json')))
    results = Parallel(n_jobs=32)(delayed(extract_authors_vocab_AMIT)(filename, authors) for filename in tqdm(filenames, desc='Reading files'))

    # merge results
    logging.info("Merging results")
    authors_vocab = ListDict()
    for r in results:
        authors_vocab.update_lists(r)

    if split_type == 'author':
        logging.info("Reading authors splits.")
        train_authors = read_splits('../data/splits/train_author.txt')
        train_authors.remove('Judgement_Bot_AITA')
    elif split_type == 'sit':
        logging.info("Loading situations splits.")
        train_situations = read_splits('../data/splits/train_sit.txt')
        train_authors = dataset.get_authors_from_situations(train_situations)
    elif split_type == 'verdicts':
        logging.info("Loading verdicts splits.")
        verdict_ids = list(dataset.verdictToLabel.keys())
        labels = list(dataset.verdictToLabel.values())
        train_verdicts, test_verdicts, train_labels, test_labels = train_test_split(verdict_ids, labels, test_size=0.2, 
                                                                            random_state=SEED)

        train_verdicts, val_verdicts, train_labels, val_labels = train_test_split(train_verdicts, train_labels, test_size=0.15, 
                                                                            random_state=SEED)
        
        train_authors = set()
        for v in train_verdicts:
            train_authors.add(dataset.verdictToAuthor[v])
            
    else:
        raise Exception("Wrong split type")
    
    verdict_embedder = pkl.load(open(os.path.join(path_to_data, 'embeddings/roberta_history.pkl'), 'rb'))
    
    # creating dicts
    authorToLabel = dict()
    for i, a in enumerate(train_authors):
        authorToLabel[a] = i
        
    verdictToAuthorLabel = dict()
    for author, vocab in authors_vocab.items():
        if author in authorToLabel:
            for v in vocab:
                verdictToAuthorLabel[v[1]] = authorToLabel[author]
    
    # create data
    embeddings = []
    verdictToId = dict()
    all_verdicts = []
    all_labels = []

    for verdict, _ in tqdm(verdictToAuthorLabel.items()):
        if verdict in verdict_embedder:
            verdictToId[verdict] = len(verdictToId)
            all_verdicts.append(verdictToId[verdict])
            all_labels.append(verdictToAuthorLabel[verdict])
            embeddings.append(torch.tensor(verdict_embedder[verdict]).unsqueeze(0))
    
    temp = torch.cat(embeddings, dim=0)
    embedding_layer = nn.Embedding.from_pretrained(temp)
    train_verdicts, val_verdicts, train_labels, val_labels = train_test_split(all_verdicts, all_labels, test_size=0.2, random_state=SEED, stratify=all_labels)
    train_dataloader = DataLoader([(train_verdicts[i], train_labels[i]) for i in range(len(train_verdicts))], batch_size=64, shuffle=True)
    val_dataloader = DataLoader([(val_verdicts[i], val_labels[i]) for i in range(len(val_verdicts))], batch_size=64)
    
    output_classes = len(authorToLabel)
    logging.info("Number of output classes {}".format(output_classes))

    model = MLPAttribution(768, 384, output_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    samples_per_class_train = get_samples_per_class(torch.tensor(train_labels))
    checkpoint_dir = os.path.join(path_to_data, f'attribution_models/{split_type}_linear_layers_{TIMESTAMP}.pt')
    num_epochs = 100
    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))
    best_accuracy = 0
    best_f1 = 0
    
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            input = batch[0]
            labels = batch[1].to(DEVICE)
            embedding = embedding_layer(input).to(DEVICE)
            output = model(embedding)
            loss = loss_fn(output, labels, samples_per_class_train, output_classes)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            
        val_metric = evaluate_mlp(model, val_dataloader)
        
        logging.info("Epoch {} **** Loss {} **** Metrics validation: {}".format(epoch, loss, val_metric))
        if val_metric['f1_score_weighted'] > best_f1:
            best_f1 = val_metric['f1_score_weighted']
            torch.save({'num_output': output_classes, 'split': split_type, 'model': model.state_dict()}, checkpoint_dir)
                