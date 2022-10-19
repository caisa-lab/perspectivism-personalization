import datetime
import string
from .clusters_utils import ListDict
from tqdm import tqdm 
import re
import emoji
from constants import DATETIME_PATTERN, NTA_KEYWORDS, YTA_KEYWORDS
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch
import json
import numpy as np


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def remove_extra_spaces(text):
    return re.sub("\s+",' ', re.sub("\s\s+",' ', text))        


def print_args(args, logger):
    for arg in vars(args):
        logger.info("{} \t \t {}".format(arg, getattr(args, arg)))
        
        
def get_samples_per_class(labels):
    return torch.bincount(labels).tolist()


def get_current_timestamp():
    return str(datetime.datetime.now()).replace(" ", "_").replace(".", ":")


def get_verdicts_labels_from_authors(dataset, authors):
    verdicts = []
    labels = []
    for a in authors:
        if a in dataset.authorsToVerdicts:
            for v in dataset.authorsToVerdicts[a]:
                if v in dataset.verdictToId: #and dataset.verdictToTokensLength[v] > 5:
                    verdicts.append(v)
                    labels.append(dataset.verdictToLabel[v])
    return verdicts, labels


def get_verdicts_labels_from_sit(dataset, situations, postToVerdicts):
    verdicts = []
    labels = []
    for s in situations:
        if s in postToVerdicts:
            for v in postToVerdicts[s]:
                verdicts.append(v)
                labels.append(dataset.verdictToLabel[v])
    return verdicts, labels


def get_and_print_metrics(gold, predictions):
    cm = confusion_matrix(gold, predictions)
    print(cm)
    f1Score_1 = f1_score(gold, predictions, average='macro')
    print("Total f1 score macro {:3f}: ".format(f1Score_1))
    f1Score_2 = f1_score(gold, predictions, average='micro')
    print("Total f1 score micro {:3f}:".format(f1Score_2))
    f1Score_3 = f1_score(gold, predictions, average='binary')
    print("Total f1 score binary {:3f}:".format(f1Score_3))
    f1Score_4 = f1_score(gold, predictions, average='weighted')
    print("Total f1 score weighted {:3f}:".format(f1Score_4))
    accuracy = accuracy_score(gold, predictions)
    print("Accuracy {:3f}:".format(accuracy))
    
    return {'macro': f1Score_1, 'micro': f1Score_2, 'binary': f1Score_3, 'weighted': f1Score_4, 'accuracy': accuracy, 'cm': cm}


def get_metrics(gold, predictions):
    return {'macro': f1_score(gold, predictions, average='macro'), 'micro': f1_score(gold, predictions, average='micro'), 
            'binary': f1_score(gold, predictions, average='binary'), 'weighted': f1_score(gold, predictions, average='weighted'), 
            'accuracy': accuracy_score(gold, predictions), 'cm': confusion_matrix(gold, predictions)}


def timestamp_to_string(timestamp):
    dt = datetime.utcfromtimestamp(timestamp)
    return dt.strftime(DATETIME_PATTERN) 

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
    
def has_link(string):
    # findall() has been used 
    # with valid conditions for urls in string
    #regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    url = re.findall(regex,string)      
    return len(url) != 0


def split_verdicts_comments_amit(author_comments):
    author_verdicts = ListDict()
    author_otherComments = ListDict()

    for author, comments in tqdm(author_comments.items(), desc="Using keywords to extract verdicts"):
        for comment, id, parent_id in comments:
            AMIT_COMMENT_FLAG = False
            MODERATOR_FLAG = False
            
            if type(comment) == dict:
                text = comment['body']
                MODERATOR_FLAG = comment['distinguished'] != 'moderator'
            else:
                text = comment
            
            if not has_link(text):
                for key in (NTA_KEYWORDS + YTA_KEYWORDS):
                    if key in text.strip().lower() and not MODERATOR_FLAG:
                        author_verdicts.append(author, (text.strip(), id, parent_id))
                        AMIT_COMMENT_FLAG = True
                        break

                if not AMIT_COMMENT_FLAG and not MODERATOR_FLAG: #and 'distinguished' in comment and comment['distinguished'] != 'moderator':
                    author_otherComments.append(author, (text.strip(), id, parent_id))
                
    return author_otherComments, author_verdicts


def clean_keywords_from_verdicts(input):
    # Preparing replacing groups
    keywords_rep = {'ampx200b': ""}
    for key in NTA_KEYWORDS + YTA_KEYWORDS:
        keywords_rep[key] = ""
    keywords_rep = dict(sorted(keywords_rep.items(), key=lambda k: len(k[0]), reverse=True))

    rep = dict((re.escape(k), v) for k, v in keywords_rep.items()) 
    pattern = re.compile("|".join(rep.keys()))
    
    if type(input) == str:
        text = pattern.sub(lambda m: rep[re.escape(m.group(0))], input.lower())
        return text.translate(str.maketrans('', '', string.punctuation))
    elif type(input) == dict:
        print("Returning cleaned dictionary. Assuming the input dictionary is of type verdicts->text")
        cleanedDict = dict()
        for verdict, text in tqdm(input.items(), desc="Removing keywords from verdicts"):
            cleanedDict[verdict] = pattern.sub(lambda m: rep[re.escape(m.group(0))], text.lower())
            cleanedDict[verdict] = cleanedDict[verdict].translate(str.maketrans('', '', string.punctuation))
    else:
        raise Exception("Wrong input type")


EMOJI_DESCRIPTION_SCRUB = re.compile(r':(\S+?):')
HASHTAG_BEFORE = re.compile(r'#(\S+)')
BAD_HASHTAG_LOGIC = re.compile(r'(\S+)!!')
FIND_MENTIONS = re.compile(r'@(\S+)')
LEADING_NAMES = re.compile(r'^\s*((?:@\S+\s*)+)')
TAIL_NAMES = re.compile(r'\s*((?:@\S+\s*)+)$')

def process_tweet(s, save_text_formatting=True, keep_emoji=False, keep_usernames=False):
    if save_text_formatting:
        s = re.sub(r'https\S+', r'', str(s))
        s = re.sub(r'http\S+', r'', str(s))
    else:
        s = re.sub(r'http\S+', r'', str(s))
        s = re.sub(r'https\S+', r' ', str(s))
        s = re.sub(r'x{3,5}', r' ', str(s))
    
    s = re.sub(r'\\n', ' ', s)
    s = re.sub(r'\s', ' ', s)
    s = re.sub(r'<br>', ' ', s)
    s = re.sub(r'&amp;', '&', s)
    s = re.sub(r'&#039;', "'", s)
    s = re.sub(r'&gt;', '>', s)
    s = re.sub(r'&lt;', '<', s)
    s = re.sub(r'\'', "'", s)

    if save_text_formatting:
        s = emoji.demojize(s)
    elif keep_emoji:
        s = emoji.demojize(s)
        s = s.replace('face_with', '')
        s = s.replace('face_', '')
        s = s.replace('_face', '')
        s = re.sub(EMOJI_DESCRIPTION_SCRUB, r' \1 ', s)
        s = s.replace('(_', '(')
        s = s.replace('_', ' ')

    s = re.sub(r"\\x[0-9a-z]{2,3,4}", "", s)
    
    if save_text_formatting:
        s = re.sub(HASHTAG_BEFORE, r'\1!!', s)
    else:
        s = re.sub(HASHTAG_BEFORE, r'\1', s)
        s = re.sub(BAD_HASHTAG_LOGIC, r'\1', s)
    
    if save_text_formatting:
        #@TODO 
        pass
    else:
        # If removing formatting, either remove all mentions, or just the @ sign.
        if keep_usernames:
            s = ' '.join(s.split())

            s = re.sub(LEADING_NAMES, r' ', s)
            s = re.sub(TAIL_NAMES, r' ', s)

            s = re.sub(FIND_MENTIONS, r'\1', s)
        else:
            s = re.sub(FIND_MENTIONS, r' ', s)
    #s = re.sub(re.compile(r'@(\S+)'), r'@', s)
    user_regex = r".?@.+?( |$)|<@mention>"    
    s = re.sub(user_regex," @user ", s, flags=re.I)
    
    # Just in case -- remove any non-ASCII and unprintable characters, apart from whitespace  
    s = "".join(x for x in s if (x.isspace() or (31 < ord(x) < 127)))
    s = ' '.join(s.split())
    return s
                
                