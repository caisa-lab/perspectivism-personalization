import json
import glob
import os
from tqdm import tqdm

from utils.utils import remove_extra_spaces
from .clusters_utils import ListDict
import pandas as pd


def read_splits(filename):
    split = []
    with open(filename, 'r') as f:
        for l in f:
            split.append(l.strip())
            
    return split


def write_splits(filename, situations):
    with open(filename, 'w') as f:
        for sit in situations:
            f.write(sit + '\n')


def read_situations(situations_path='/app/public/joan/submissions_of_amit_filtered_history'):
    filenames = sorted(glob.glob(os.path.join(situations_path, '*.json')))

    situations_dataset = {'author_name': [], 'id': [], 
                'text': [], 'title': [], 'label': [], 
                'ups': [], 'downs': [], 'upvote_ratio': []}


    for filename in tqdm(filenames):
        with open(filename, 'r') as in_file:
            for content in in_file:
                situations = json.loads(content)
                for situation in situations:
                    if 'author_fullname' in situation:
                        situations_dataset['author_name'].append(situation['author_fullname'])
                    else:
                        situations_dataset['author_name'].append('')

                    situations_dataset['id'].append(situation['id'])
                    situations_dataset['text'].append(situation['selftext'])
                    situations_dataset['title'].append(situation['title'])
                    situations_dataset['label'].append(situation['link_flair_text'])
                    situations_dataset['ups'].append(situation['ups'])
                    situations_dataset['downs'].append(situation['downs'])
                    situations_dataset['upvote_ratio'].append(situation['upvote_ratio'])
                    
            
    assert len(situations_dataset['author_name']) == len(situations_dataset['id']) and len(situations_dataset['author_name']) == len(situations_dataset['text']) and \
    len(situations_dataset['author_name']) == len(situations_dataset['title']) and len(situations_dataset['author_name']) == len(situations_dataset['label']) and \
    len(situations_dataset['author_name']) == len(situations_dataset['ups']) and len(situations_dataset['author_name']) == len(situations_dataset['downs']) and \
    len(situations_dataset['author_name']) == len(situations_dataset['upvote_ratio']) 

    return situations_dataset


def read_authors(age_dir = '../data/demographic/age_list.csv', gender_dir = '../data/demographic/gender_list.csv'):
    age_df = pd.read_csv(age_dir, sep='\t', names=['author', 'subreddit', 'age'])
    gender_df = pd.read_csv(gender_dir, sep='\t', names=['author', 'gender'])

    age_authors = set(age_df.author)
    gender_authors = set(gender_df.author)
    authors = gender_authors.intersection(age_authors)

    return authors
    

def read_amit_json_files(dirname = '../data/amit_filtered_history'):
    filenames = sorted(glob.glob(os.path.join(dirname, '*.json')))
    file_comments = ListDict()

    for filename in tqdm(filenames, desc='Reading files'):
        key = filename.split('/')[-1].split('.')[0]
        with open(filename, 'r') as in_file:
            file_comments[key] = json.load(in_file)
            
    return file_comments
                    
                
def read_file_txt(filename):
    file_comments = ListDict()

    key = filename.split('/')[-1].split('_')[-1]
     
    with open(filename, 'r') as in_file:
        for comment in in_file:
            #file_comments.append(key, json.loads(comment))
            file_comments.append(key, comment)
                
    return file_comments


def extract_authors_vocab_notAMIT(filename, authors):
    author_comments = ListDict()
     
    with open(filename, 'r') as in_file:
        for comment in in_file:
            comment = json.loads(comment)
            
            if comment['author'] in authors and comment['subreddit'].lower() != 'amitheasshole':
                author_comments.append(comment['author'], (comment['body'], comment['created_utc']))
                            
    return author_comments


def extract_authors_vocab_fullHistory(filename, authors):
    author_comments = ListDict()
     
    with open(filename, 'r') as in_file:
        for comment in in_file:
            comment = json.loads(comment)
            
            if comment['author'] in authors:
                author_comments.append(comment['author'], (comment['body'], comment['created_utc'], comment['subreddit']))
                            
    return author_comments

def extract_authors_subreddits(filename, authors):
    author_subreddits = ListDict()
     
    with open(filename, 'r') as in_file:
        for comment in in_file:
            comment = json.loads(comment)
            
            if comment['author'] in authors:
                author_subreddits.append(comment['author'], comment['subreddit'])
                            
    return author_subreddits

def extract_authors_vocab_AMIT(filename, authors):
    author_comments = ListDict()

    #key = filename.split('/')[-1].split('.')[0]
    with open(filename, 'r') as in_file:
        comments = json.load(in_file)
        for comment in comments:
            if comment['author'] in authors:
                author_comments.append(comment['author'], (comment['body'], comment['id'], comment['parent_id']))
                    
    return author_comments


def write_persona(path, vocabTimes, nlp, keywords= ['my', 'myself']):
    vocab, times, subreddit = zip(*vocabTimes)

    sentences = []
    docs = nlp.pipe(vocab, n_process=8, batch_size=5000)
    for i, doc in enumerate(docs): 
        for sent in doc.sents:
            if (sent.text.startswith('i ') or any(key in sent.text for key in keywords)) and len(sent) > 5:
                sentences.append((sent.text, times[i], subreddit[i]))
     
    
    with open(path, 'w') as file:
        
        for sent, time, subreddit in sentences:
            file.write(remove_extra_spaces(sent) + '\t' + str(time) + '\t' + str(subreddit) + '\n')
            

def read_persona(dirname, authors):
    filenames = sorted(glob.glob(os.path.join(dirname, '*')))
    authors_personas = ListDict()
    for filename in tqdm(filenames):
        key = filename.split('/')[-1].split('.txt')[0]
        if key in authors:
            with open(filename, 'r') as file:
                for line in file.readlines():
                    temp = line.split('\t')
                    assert len(temp) == 3
                    authors_personas.append(key, temp[0])    
        else:
            print(key, " not in the authors list")
    
    return authors_personas  
