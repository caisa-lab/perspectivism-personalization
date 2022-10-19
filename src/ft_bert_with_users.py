import glob
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertTokenizerFast, DataCollatorWithPadding, AdamW, get_scheduler
from datasets import DatasetDict, Dataset, Features, Value
from torch_geometric.data import Data


from dataset import GraphData, SocialNormDataset, VerdictDataset
from utils.read_files import *
from utils.utils import *
from utils.loss_functions import *
from utils.train_utils import *
from models import GAT, JudgeBert, SentBertClassifier
from constants import *
from tqdm.auto import tqdm
from argparse import ArgumentParser
import logging

TIMESTAMP = get_current_timestamp()

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
    
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"../logs/{TIMESTAMP}.log"),
        logging.StreamHandler()
    ]
)

parser = ArgumentParser()

parser.add_argument("--path_to_data", dest="path_to_data", required=True, type=str)
parser.add_argument("--use_authors", dest="use_authors", required=True, type=str2bool)
parser.add_argument("--author_encoder", dest="author_encoder", required=True, type=str) # ['average', 'priming', 'graph', 'none']

parser.add_argument("--split_type", dest="split_type", required=True, type=str) # ['author', 'sit', 'verdicts']
parser.add_argument("--sbert_model", dest="sbert_model", default='sentence-transformers/all-distilroberta-v1', type=str)
parser.add_argument("--authors_embedding_path", dest="authors_embedding_path", required=True, type=str)
parser.add_argument("--sbert_dim", dest="sbert_dim", default=768, type=int)
parser.add_argument("--user_dim", dest="user_dim", default=768, type=int)
parser.add_argument("--graph_dim", dest="graph_dim", default=384, type=int)
parser.add_argument("--concat", dest="concat", default='true', type=str2bool)
parser.add_argument("--num_epochs", dest="num_epochs", default=5, type=int)
parser.add_argument("--learning_rate", dest="learning_rate", default=1e-4, type=float)
parser.add_argument("--batch_size", dest="batch_size", default=32, type=int)
parser.add_argument("--loss_type", dest="loss_type", default='softmax', type=str)
parser.add_argument("--verdicts_dir", dest="verdicts_dir", default='../data/verdicts', type=str)
parser.add_argument("--bert_tok", dest="bert_tok", default='bert-base-uncased', type=str)
parser.add_argument("--dirname", dest="dirname", type=str, default='../data/amit_filtered_history')
parser.add_argument("--results_dir", dest="results_dir", type=str, default='../results')
parser.add_argument("--model_name", dest="model_name", type=str, required=True) # ['judge_bert', 'sbert'] otherwise exception


if __name__ == '__main__':
    args = parser.parse_args()
    print_args(args, logging)
    path_to_data = args.path_to_data

    dirname = args.dirname
    bert_checkpoint = args.bert_tok
    model_name = args.model_name
    results_dir = args.results_dir
    verdicts_dir = args.verdicts_dir
    graph_dim = args.graph_dim
    checkpoint_dir = os.path.join(results_dir, f'best_models/{TIMESTAMP}_best_model_sampled.pt')
    graph_checkpoint_dir = os.path.join(results_dir, f'best_models/{TIMESTAMP}_best_graphmodel.pt')
    authors_embedding_path = args.authors_embedding_path
    USE_AUTHORS = args.use_authors
    author_encoder = args.author_encoder
    if USE_AUTHORS:
        assert author_encoder in {'average', 'graph', 'attribution'}
    else:
        assert author_encoder.lower() == 'none' or author_encoder.lower() == 'priming' or author_encoder.lower() == 'user_id'
        
    split_type = args.split_type
    
    
    logging.info("Device {}".format(DEVICE))
    
    social_norm = True #@TODO: Fixed for EMNLP, should make it a parameter
    if social_norm:
        path_to_data = '/app/public/joan/' 
        social_chemistry = pd.read_pickle(path_to_data +'social_chemistry_clean_with_fulltexts_and_authors.gzip', compression='gzip')

        path_to_data = '/data/joan/users_perception/data_bela/'
        with open(path_to_data+'social_norms_clean.csv') as file:
            social_comments = pd.read_csv(file)
        
        dataset = SocialNormDataset(social_comments, social_chemistry)
    else:
        logging.info(f'Processing json files from directory {dirname}')
        authors = read_authors()
        filenames = sorted(glob.glob(os.path.join(dirname, '*.json')))
        results = Parallel(n_jobs=32)(delayed(extract_authors_vocab_AMIT)(filename, authors) for filename in tqdm(filenames, desc='Reading files'))
        
        authors_vocab = ListDict()
        for r in results:
            authors_vocab.update_lists(r)
            
        dataset = VerdictDataset(authors_vocab)
    
    
    if split_type == 'sit':
        logging.info("Split type {}".format(split_type))
        train_verdicts, train_labels, val_verdicts, val_labels, test_verdicts, test_labels = get_verdicts_by_situations_split(dataset)
    elif split_type == 'author':
        logging.info("Split type {}".format(split_type))
        train_verdicts, train_labels, val_verdicts, val_labels, test_verdicts, test_labels = get_verdicts_by_author_split(dataset)
    elif split_type == 'verdicts':
        logging.info("Split type {}".format(split_type))
        verdict_ids = list(dataset.verdictToLabel.keys())
        labels = list(dataset.verdictToLabel.values())
        train_verdicts, test_verdicts, train_labels, test_labels = train_test_split(verdict_ids, labels, test_size=0.2, 
                                                                            random_state=SEED)

        train_verdicts, val_verdicts, train_labels, val_labels = train_test_split(train_verdicts, train_labels, test_size=0.15, 
                                                                            random_state=SEED)
    else:
        raise Exception("Split type is wrong, it should be either sit or author")    
   
    
    train_size_stats = "Training Size: {}, NTA labels {}, YTA labels {}".format(len(train_verdicts), train_labels.count(0), train_labels.count(1))
    logging.info(train_size_stats)
    val_size_stats = "Validation Size: {}, NTA labels {}, YTA labels {}".format(len(val_verdicts), val_labels.count(0), val_labels.count(1))
    logging.info(val_size_stats)
    test_size_stats = "Test Size: {}, NTA labels {}, YTA labels {}".format(len(test_verdicts), test_labels.count(0), test_labels.count(1))
    logging.info(test_size_stats)
    
    if author_encoder == 'priming':
        authorToSampledText = pkl.load(open('../data/priming_text.pkl', 'rb'))
    
    graph_model = None
    data = None
    if USE_AUTHORS and (author_encoder == 'average' or author_encoder == 'attribution'):
        embedder = AuthorsEmbedder(embeddings_path=authors_embedding_path, dim=args.user_dim)
    elif USE_AUTHORS and author_encoder == 'graph':
        logging.info("Creating graph")
        embedder = pkl.load(open('../data/embeddings/emnlp/sbert_authorAMIT.pkl', 'rb'))
        authorToAuthor = json.load(open('../data/authors_interactions_in_history.json', 'r'))
        graphData, edge_index = create_author_graph(GraphData(), dataset, embedder, authorToAuthor, limit_connections=100)
        data = Data(x=torch.stack(graphData.representations), edge_index=edge_index.contiguous())
        
        if args.concat:
            graph_model = GAT(graph_dim, graph_dim, dropout=0.2, heads=2, concat=True).to(DEVICE)
        else:
            graph_model = GAT(graph_dim, graph_dim, dropout=0.2, heads=2, concat=False).to(DEVICE)

    else:
        embedder = None
    
    
    raw_dataset = {'train': {'index': [], 'text': [], 'label': [], 'author_node_idx': []}, 
            'val': {'index': [], 'text': [], 'label': [], 'author_node_idx': []}, 
            'test': {'index': [], 'text': [], 'label': [], 'author_node_idx': [] }}

    
    for i, verdict in enumerate(train_verdicts):
        situation_title = dataset.postIdToTitle[dataset.verdictToParent[verdict]]
        if situation_title != '' and situation_title is not None and verdict in dataset.verdictToAuthor:
            author = dataset.verdictToAuthor[verdict]
            
            if author != 'Judgement_Bot_AITA':
                raw_dataset['train']['index'].append(dataset.verdictToId[verdict])
                
                if author_encoder == 'priming':
                    author = dataset.verdictToAuthor[verdict]
                    priming_text = ''
                    if author in authorToSampledText:
                        priming_text = authorToSampledText[author]
                    raw_dataset['train']['text'].append(priming_text + ' [SEP] ' + situation_title + ' [SEP] ' + dataset.verdictToCleanedText[verdict])
                elif author_encoder == 'user_id':
                    author = dataset.verdictToAuthor[verdict]
                    raw_dataset['train']['text'].append('[' + author + ']' + ' [SEP] ' + situation_title + ' [SEP] ' + dataset.verdictToCleanedText[verdict])
                else:
                    raw_dataset['train']['text'].append(situation_title + ' [SEP] ' + dataset.verdictToCleanedText[verdict])
                    
                raw_dataset['train']['label'].append(train_labels[i])
                
                if USE_AUTHORS and author_encoder == 'graph':
                    raw_dataset['train']['author_node_idx'].append(graphData.nodesToId[dataset.verdictToAuthor[verdict]])
                else:
                    raw_dataset['train']['author_node_idx'].append(-1)
                    
                assert train_labels[i] == dataset.verdictToLabel[verdict] 
        
    for i, verdict in enumerate(val_verdicts):
        situation_title = dataset.postIdToTitle[dataset.verdictToParent[verdict]]
        if situation_title != '' and situation_title is not None and verdict in dataset.verdictToAuthor:
            author = dataset.verdictToAuthor[verdict]
            
            if author != 'Judgement_Bot_AITA': 
                raw_dataset['val']['index'].append(dataset.verdictToId[verdict])
                # Priming logic
                if author_encoder == 'priming':
                    author = dataset.verdictToAuthor[verdict]
                    priming_text = ''
                    if author in authorToSampledText:
                        priming_text = authorToSampledText[author]
                    raw_dataset['val']['text'].append(priming_text + ' [SEP] ' + situation_title + ' [SEP] ' + dataset.verdictToCleanedText[verdict])
                elif author_encoder == 'user_id':
                    author = dataset.verdictToAuthor[verdict]
                    raw_dataset['val']['text'].append('[' + author + ']' + ' [SEP] ' + situation_title + ' [SEP] ' + dataset.verdictToCleanedText[verdict])
                else:
                    raw_dataset['val']['text'].append(situation_title + ' [SEP] ' + dataset.verdictToCleanedText[verdict])
                
                raw_dataset['val']['label'].append(val_labels[i])
                
                if USE_AUTHORS and author_encoder == 'graph':
                    raw_dataset['val']['author_node_idx'].append(graphData.nodesToId[dataset.verdictToAuthor[verdict]])
                else:
                    raw_dataset['val']['author_node_idx'].append(-1)
                
                assert val_labels[i] == dataset.verdictToLabel[verdict]          
        
    for i, verdict in enumerate(test_verdicts):
        situation_title = dataset.postIdToTitle[dataset.verdictToParent[verdict]]
        if situation_title != '' and situation_title is not None and verdict in dataset.verdictToAuthor:
            author = dataset.verdictToAuthor[verdict]
            
            if author != 'Judgement_Bot_AITA': 
                raw_dataset['test']['index'].append(dataset.verdictToId[verdict])
                # Priming logic
                if author_encoder == 'priming':
                    author = dataset.verdictToAuthor[verdict]
                    priming_text = ''
                    if author in authorToSampledText:
                        priming_text = authorToSampledText[author]
                    # priming text contains the [SEP] in the end by itself
                    raw_dataset['test']['text'].append(priming_text + ' [SEP] ' + situation_title + ' [SEP] ' + dataset.verdictToCleanedText[verdict])
                elif author_encoder == 'user_id':
                    author = dataset.verdictToAuthor[verdict]
                    raw_dataset['test']['text'].append('[' + author + ']' + ' [SEP] ' + situation_title + ' [SEP] ' + dataset.verdictToCleanedText[verdict])
                else:
                    raw_dataset['test']['text'].append(situation_title + ' [SEP] ' + dataset.verdictToCleanedText[verdict])
                    
                raw_dataset['test']['label'].append(test_labels[i])
                
                if USE_AUTHORS and author_encoder == 'graph':   
                    raw_dataset['test']['author_node_idx'].append(graphData.nodesToId[dataset.verdictToAuthor[verdict]])
                else:
                    raw_dataset['test']['author_node_idx'].append(-1)
                
                assert test_labels[i] == dataset.verdictToLabel[verdict] 
    

    if model_name == 'sbert':
        logging.info("Training with SBERT, model name is {}".format(model_name))
        tokenizer = AutoTokenizer.from_pretrained(bert_checkpoint)
        model = SentBertClassifier(users_layer=USE_AUTHORS, user_dim=args.user_dim, sbert_model=args.sbert_model, sbert_dim=args.sbert_dim)
    elif model_name == 'judge_bert':
        logging.info("Training with Judge Bert, model name is {}".format(model_name))
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        model = JudgeBert()
    else:
        raise Exception('Wrong model name')
    
    
    model.to(DEVICE)
    
    ds = DatasetDict()

    for split, d in raw_dataset.items():
        ds[split] = Dataset.from_dict(mapping=d, features=Features({'label': Value(dtype='int64'), 
                                                                        'text': Value(dtype='string'), 'index': Value(dtype='int64'), 'author_node_idx': Value(dtype='int64')}))
    
    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True)

    logging.info("Tokenizing the dataset")
    tokenized_dataset = ds.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")
    
    batch_size = args.batch_size
    #class_sample_count = [train_labels.count(0), train_labels.count(1)]
    #weights = 1 / torch.Tensor(class_sample_count)
    #sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_size)
    
    train_dataloader = DataLoader(
        tokenized_dataset["train"], batch_size=batch_size, collate_fn=data_collator, shuffle = True
    )
    eval_dataloader = DataLoader(
        tokenized_dataset["val"], batch_size=batch_size, collate_fn=data_collator
    )
    
    test_dataloader = DataLoader(
        tokenized_dataset["test"], batch_size=batch_size, collate_fn=data_collator
    )

    if USE_AUTHORS and author_encoder == 'graph':
        logging.info("Grouping parameters")
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters()], 'weight_decay': 0.01},
            {'params': [p for n, p in graph_model.named_parameters()], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    else:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    num_epochs = args.num_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    samples_per_class_train = get_samples_per_class(tokenized_dataset["train"]['labels'])

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    logging.info("Number of training steps {}".format(num_training_steps))
    loss_type=args.loss_type
    progress_bar = tqdm(range(num_training_steps))
    best_accuracy = 0
    best_f1 = 0
    val_metrics = []
    train_loss = []
    
    
    for epoch in range(num_epochs):
        model.train()
        if USE_AUTHORS and author_encoder == 'graph': 
            graph_model.train()
        for batch in train_dataloader:
            verdicts_index = batch.pop("index")
            author_node_idx = batch.pop("author_node_idx")
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = batch.pop("labels")
            
            if USE_AUTHORS and  (author_encoder == 'average' or author_encoder == 'attribution'):
                authors_embeddings = torch.stack([embedder.embed_author(dataset.verdictToAuthor[dataset.idToVerdict[index.item()]]) for index in verdicts_index]).to(DEVICE)
                output = model(batch, authors_embeddings)
            elif USE_AUTHORS and author_encoder == 'graph':
                graph_output = graph_model(data.x.to(DEVICE), data.edge_index.to(DEVICE))
                authors_embeddings = graph_output[author_node_idx.to(DEVICE)]
                output = model(batch, authors_embeddings)
            else:
                output = model(batch)
            
            loss = loss_fn(output, labels, samples_per_class_train, loss_type=loss_type)
            train_loss.append(loss.item())
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        
        val_metric = evaluate(eval_dataloader, model, graph_model, data, embedder, USE_AUTHORS, dataset, author_encoder)
        val_metrics.append(val_metric)
        
        logging.info("Epoch {} **** Loss {} **** Metrics validation: {}".format(epoch, loss, val_metric))
        if val_metric['f1_weighted'] > best_f1:
            best_f1 = val_metric['f1_weighted']
            torch.save(model.state_dict(), checkpoint_dir)
            if USE_AUTHORS and author_encoder == 'graph':
                torch.save(graph_model.state_dict(), graph_checkpoint_dir)        
        
              

    logging.info("Evaluating")
    model.load_state_dict(torch.load(checkpoint_dir))
    model.to(DEVICE)
    if USE_AUTHORS and author_encoder == 'graph':
        graph_model.load_state_dict(torch.load(graph_checkpoint_dir))
        graph_model.to(DEVICE)
        
    test_metrics = evaluate(test_dataloader, model, graph_model, data, embedder, USE_AUTHORS, dataset, author_encoder, True)
    results = test_metrics.pop('results')
    logging.info(test_metrics)
    
    result_logs = {'id': TIMESTAMP}
    result_logs['seed'] = SEED
    result_logs['sbert_model'] = args.sbert_model
    result_logs['model_name'] = args.model_name
    result_logs['use_authors_embeddings'] = USE_AUTHORS
    result_logs['authors_embedding_path'] = authors_embedding_path
    result_logs['author_encoder'] = author_encoder
    result_logs['split_type'] = split_type
    result_logs['train_stats'] = train_size_stats
    result_logs['val_stats'] = val_size_stats
    result_logs['test_stats'] = test_size_stats
    result_logs['epochs'] = num_epochs
    result_logs['optimizer'] = optimizer.defaults
    result_logs["loss_type"] = loss_type
    result_logs['test_metrics'] = test_metrics
    result_logs['checkpoint_dir'] = checkpoint_dir
    result_logs['val_metrics'] = val_metrics
    result_logs['results'] = results
    
    
    res_file = os.path.join(results_dir, TIMESTAMP + ".json")
    with open(res_file, mode='w') as f:
        json.dump(result_logs, f, cls=NpEncoder, indent=2)

    

        
    
    

    