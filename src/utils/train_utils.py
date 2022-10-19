import json
import os
from datasets import load_metric
from numpy import average
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from utils.clusters_utils import ListDict
from utils.loss_functions import CB_loss
from constants import DEVICE
import pickle as pkl
from utils.read_files import read_splits, write_splits
from utils.utils import get_verdicts_labels_from_sit, get_verdicts_labels_from_authors
from constants import SEED


class AuthorsEmbedder:
    def __init__(self, embeddings_path, dim):
        self.authors_embeddings = pkl.load(open(embeddings_path, 'rb'))
        self.dim = dim
    
    
    def embed_author(self, author):
        return torch.tensor(self.authors_embeddings.get(author, torch.rand(self.dim)))
        
    

# class AuthorsEmbedder:
#     def __init__(self, amit_embeddings_path='../data/embeddings/emnlp/sbert_authorAMIT.pkl', 
#                  no_amit_embeddings_path='../data/embeddings/emnlp/sbert_authorNotAMIT.pkl', 
#                  only_amit=False, only_no_amit=False, dim=768):
#         self.only_amit = only_amit
#         self.only_no_amit = only_no_amit
#         self.dim = dim
    
#         self.authorAMIT_embeddings = pkl.load(open(amit_embeddings_path, 'rb'))
#         self.authorNotAMIT_embeddings = pkl.load(open(no_amit_embeddings_path, 'rb'))
    
    
#     def embed_author(self, author):
#         if self.only_amit:
#             return self.authorAMIT_embeddings.get(author, torch.rand(self.dim))
        
#         if self.only_no_amit:
#             return self.authorAMIT_embeddings.get(author, torch.rand(self.dim))
        
#         if author in self.authorAMIT_embeddings and author not in self.authorNotAMIT_embeddings:
#             return self.authorAMIT_embeddings[author]
#         elif author in self.authorNotAMIT_embeddings and author not in self.authorAMIT_embeddings:
#             return self.authorNotAMIT_embeddings[author]
#         else:
#             amit_embeddings = self.authorAMIT_embeddings[author]
#             noamit_embeddings = self.authorNotAMIT_embeddings[author]
#             embeddings = torch.cat([amit_embeddings.unsqueeze(0), noamit_embeddings.unsqueeze(0)], dim=0)
#             return torch.mean(embeddings, dim=0)
        

def loss_fn(output, targets, samples_per_cls, no_of_classes=2, loss_type = "softmax"):
    beta = 0.9999
    gamma = 2.0

    return CB_loss(targets, output, samples_per_cls, no_of_classes, loss_type, beta, gamma)


def get_verdicts_by_situations_split(dataset):
    if not os.path.exists('../data/splits/train_sit.txt'):
        all_situations = set(dataset.postIdToId.keys())
        annotated_situations = json.load(open('../data/conflict_aspect_annotations.json', 'r'))
        annotated_situations = set(annotated_situations['data'].keys())
        all_situations = list(all_situations.difference(annotated_situations))

        train_situations, test_situations = train_test_split(all_situations, test_size=0.18, random_state=SEED)
        train_situations, val_situations = train_test_split(train_situations, test_size=0.15, random_state=SEED)
        test_situations.extend(list(annotated_situations))
        write_splits('../data/splits/train_sit.txt', train_situations)
        write_splits('../data/splits/test_sit.txt', test_situations)
        write_splits('../data/splits/val_sit.txt', val_situations)
    else:
        print("Loading situations splits.")
        train_situations = read_splits('../data/splits/train_sit.txt')
        val_situations = read_splits('../data/splits/val_sit.txt')
        test_situations = read_splits('../data/splits/test_sit.txt')
        
    postToVerdicts = ListDict()
    for v, s in dataset.verdictToParent.items():
        #if dataset.verdictToTokensLength[v] > 5:
        postToVerdicts.append(s, v)
        
    train_verdicts, train_labels = get_verdicts_labels_from_sit(dataset, train_situations, postToVerdicts)
    val_verdicts, val_labels = get_verdicts_labels_from_sit(dataset, val_situations, postToVerdicts)
    test_verdicts, test_labels = get_verdicts_labels_from_sit(dataset, test_situations, postToVerdicts)
    return train_verdicts, train_labels, val_verdicts, val_labels, test_verdicts, test_labels


def get_verdicts_by_author_split(dataset):
    if not os.path.exists('../data/splits/train_author.txt'):
            all_authors = list(dataset.authorsToVerdicts.keys())
            train_authors, test_authors = train_test_split(all_authors, test_size=0.2, random_state=SEED)
            train_authors, val_authors = train_test_split(train_authors, test_size=0.14, random_state=SEED)
            write_splits('../data/splits/train_author.txt', train_authors)
            write_splits('../data/splits/val_author.txt', val_authors)
            write_splits('../data/splits/test_author.txt', test_authors)
    else:
        print("Reading authors splits.")
        train_authors = read_splits('../data/splits/train_author.txt')
        val_authors = read_splits('../data/splits/val_author.txt')
        test_authors = read_splits('../data/splits/test_author.txt')
        # train_authors.remove('Judgement_Bot_AITA')
        
    train_verdicts, train_labels = get_verdicts_labels_from_authors(dataset, train_authors)
    val_verdicts, val_labels = get_verdicts_labels_from_authors(dataset, val_authors)
    test_verdicts, test_labels = get_verdicts_labels_from_authors(dataset, test_authors)
    return train_verdicts, train_labels, val_verdicts, val_labels, test_verdicts, test_labels


def evaluate(dataloader, model, graph_model, data, embedder, USE_AUTHORS, dataset, author_encoder, return_predictions=False):
    accuracy_metric = load_metric("accuracy")
    f1_metric = load_metric("f1")
    
    model.eval()
    if USE_AUTHORS and author_encoder == 'graph': 
        graph_model.eval()
        
    all_ids = ['verdicts']
    all_pred = ['predictions']
    all_labels = ['gold labels']
    
    for batch in dataloader:
        verdicts_index = batch.pop("index")
        author_node_idx = batch.pop("author_node_idx")
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        labels = batch.pop("labels")
        with torch.no_grad():
            if USE_AUTHORS and  (author_encoder == 'average' or author_encoder == 'attribution'):
                authors_embeddings =  torch.stack([embedder.embed_author(dataset.verdictToAuthor[dataset.idToVerdict[index.item()]]) for index in verdicts_index]).to(DEVICE)
                logits = model(batch, authors_embeddings)
            elif USE_AUTHORS and author_encoder == 'graph':
                graph_output = graph_model(data.x.to(DEVICE), data.edge_index.to(DEVICE))
                authors_embeddings = graph_output[author_node_idx.to(DEVICE)]
                logits = model(batch, authors_embeddings)
            else:
                logits = model(batch)

        predictions = torch.argmax(logits, dim=-1)
        accuracy_metric.add_batch(predictions=predictions, references=labels)
        f1_metric.add_batch(predictions=predictions, references=labels)
        all_pred.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_ids.extend([dataset.idToVerdict[idx] for idx in verdicts_index.numpy()])
    
    
    if return_predictions:
        return {'accuracy': accuracy_metric.compute()['accuracy'], 'f1_weighted': f1_metric.compute(average='weighted')['f1'], 
                'macro': f1_score(all_labels[1:], all_pred[1:], average='macro'),
                'micro': f1_score(all_labels[1:], all_pred[1:], average='micro'), 
                'binary': f1_score(all_labels[1:], all_pred[1:], average='binary'),
                'results': list(zip(all_ids, all_pred, all_labels))}

    return {'accuracy': accuracy_metric.compute()['accuracy'], 'f1_weighted': f1_metric.compute(average='weighted')['f1'],
             'macro': f1_score(all_labels[1:], all_pred[1:], average='macro'),
                'micro': f1_score(all_labels[1:], all_pred[1:], average='micro'), 
                'binary': f1_score(all_labels[1:], all_pred[1:], average='binary')}


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def create_author_graph(graphData, dataset, authors_embeddings, authorToAuthor, limit_connections=100):
    leave_out = {'Judgement_Bot_AITA'}
    for author, _ in dataset.authorsToVerdicts.items():
        if author not in leave_out:
            graphData.addNode(author, 'author', authors_embeddings[author], None, None)
            
    # Add author to author edges
    source = []
    target = []
    for author, neighbors in tqdm(authorToAuthor.items()):
        neighbors.sort(key=lambda x: x[1], reverse=True)
        if len(neighbors) > limit_connections:
            neighbors = neighbors[:limit_connections]
            
        for neighbor in neighbors:
            # neighbor[0] = author, neighbor[1] = number_of_connections
            if author in graphData.nodesToId and neighbor[0] in graphData.nodesToId:
                source.append(graphData.nodesToId[author])
                target.append(graphData.nodesToId[neighbor[0]])
            
    
    return graphData, torch.tensor([source, target], dtype=torch.long)