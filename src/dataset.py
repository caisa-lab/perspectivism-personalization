import re
from tqdm import tqdm
from constants import NTA_KEYWORDS, YTA_KEYWORDS
from utils.clusters_utils import ListDict
from utils.utils import remove_extra_spaces, split_verdicts_comments_amit
import os
import pickle as pkl
import string
from abc import ABC, abstractmethod
import torch 


class GraphData:
    def __init__(self):
        self.nodesToId = {}
        self.idToNodes = []
        self.labels = []
        self.labelsToId = {'verdict': 0, 'situation': 2, 'author': 3}
        self.verdicts_mask = []
        self.representations = []
        self.train_mask = []
        self.val_mask = []
        self.test_mask = []
        
        self.train_verdicts = []
        self.val_verdicts = []
        self.test_verdicts = []
        
        self.train_users = []
        self.val_users = []
        self.test_users = []
        
        self.train_sit = []
        self.val_sit = []
        self.test_sit = []
    
    def addNode(self, node, name, representation, split, label):
        if node not in self.nodesToId:
            self.nodesToId[node] = len(self.idToNodes)
            self.idToNodes.append(node)
            
            self.representations.append(torch.tensor(representation))
            if name == 'verdict':
                self.verdicts_mask.append(1)
                self.labels.append(label)
                if split == 'train':
                    self.train_mask.append(1)
                    self.val_mask.append(0)
                    self.test_mask.append(0)
                elif split == 'val':
                    self.train_mask.append(0)
                    self.val_mask.append(1)
                    self.test_mask.append(0)
                elif split == 'test':
                    self.train_mask.append(0)
                    self.val_mask.append(0)
                    self.test_mask.append(1)
                else:
                    raise Exception("Wrong split name")
            else: # it means it is an author or situation, should be masked out
                self.verdicts_mask.append(0)
                self.labels.append(self.labelsToId[name])
                self.train_mask.append(0)
                self.val_mask.append(0)
                self.test_mask.append(0)
    
    def extract_all_ids(self, dataset):
        for i, m in enumerate(self.train_mask):
            if m:
                v = self.idToNodes[i]
                a = dataset.verdictToAuthor[v]
                s = dataset.verdictToParent[v]
                self.train_verdicts.append(i)
                self.train_users.append(self.nodesToId[a])
                self.train_sit.append(self.nodesToId[s])
                
        for i, m in enumerate(self.val_mask):
            if m:
                v = self.idToNodes[i]
                a = dataset.verdictToAuthor[v]
                s = dataset.verdictToParent[v]
                self.val_verdicts.append(i)
                self.val_users.append(self.nodesToId[a])
                self.val_sit.append(self.nodesToId[s])


        for i, m in enumerate(self.test_mask):
            if m:
                v = self.idToNodes[i]
                a = dataset.verdictToAuthor[v]
                s = dataset.verdictToParent[v]
                self.test_verdicts.append(i)
                self.test_users.append(self.nodesToId[a])
                self.test_sit.append(self.nodesToId[s])
                          
                
class Dataset(ABC):
    def __init__(self):
        self.verdictToId = {}
        self.idToVerdict = []
        self.verdictToText = {}
        self.verdictToParent = {}
        self.verdictToCleanedText = {}
        self.verdictToLabel = {}
        self.postIdToTitle = {}
        self.postIdToText = {}
        self.idTopostId = []
        self.postIdToId = {}
        self.authorsToVerdicts = ListDict()
        self.verdictToAuthor = {}
        self.aita_labels = ['NTA','YTA']

    @abstractmethod
    def load_maps(self, sc, sn):
        pass
    
class SocialNormDataset:
    """Creates the verdicts datastructures to use through out the project
    """
    def __init__(self, sc, sn, cond=5):
        self.verdictToId = {}
        self.idToVerdict = []
        self.verdictToText = {}
        self.verdictToParent = {}
        self.verdictToCleanedText = {}
        self.verdictToLabel = {}
        self.verdictToTokensLength = {}
        self.postIdToTitle = {}
        self.postIdToText = {}
        self.idTopostId = []
        self.postIdToId = {}
        self.filtering_cond = cond
        self.authorsToVerdicts = ListDict()
        self.verdictToAuthor = {}
        self.postToVerdicts = ListDict()
        #self.aita_labels = ['NTA','YTA','NAH','ESH','INFO']
        self.aita_labels = ['NTA','YTA']
        

        self.load_maps(sc, sn)


    def load_maps(self, sc, sn):
        for _, row in tqdm(sn.iterrows(), desc='Creating situations maps'):
            post_id = row['post_id']
            self.postIdToTitle[post_id] = row['situation']
            self.postIdToText[post_id] = row['fulltext']
            self.idTopostId.append(post_id)
            self.postIdToId[post_id] = len(self.idTopostId) - 1
        
        for _, row in tqdm(sc.dropna(subset=['author_name', 'author_fullname']).iterrows(), desc="Creating filtered verdict-authors maps by condition {}".format(self.filtering_cond)):
            label = row['label']
            verdict = row['id']
            if label in self.aita_labels and row['author_name'] != 'Judgement_Bot_AITA':
                self.verdictToAuthor[verdict] = row['author_name']    
                self.authorsToVerdicts.append(row['author_name'], verdict)

        authorsToCount = {k: len(v) for k, v in self.authorsToVerdicts.items()}
        filtering_cond = 5

        for a, c in authorsToCount.items():
            if c <= filtering_cond:
                verdicts = self.authorsToVerdicts.pop(a)
                for v in verdicts:
                    del self.verdictToAuthor[v]         
                    
        print("After filtering, we are left with {} authors and {} verdicts.".format(len(self.authorsToVerdicts), len(self.verdictToAuthor)))   
        
        for _, row in tqdm(sc.iterrows(), desc="Creating comments maps"):
            label = row['label']
            parent = row['parent_id']
            verdict = row['id']
            
            if label in self.aita_labels and parent in self.postIdToId and verdict in self.verdictToAuthor:
                text = row['body']
                self.idToVerdict.append(verdict)
                self.verdictToId[verdict] = len(self.idToVerdict) - 1
                self.verdictToText[verdict] = text
                self.verdictToLabel[verdict] = self.aita_labels.index(label)
                self.verdictToParent[verdict] = parent
        
        self._clean_keywords_from_verdicts()
        self.verdictToTokensLength = {k: len(v.split(' ')) for k, v in self.verdictToCleanedText.items()}
        
        for v, s in self.verdictToParent.items():
            self.postToVerdicts.append(s, v)

    def _clean_keywords_from_verdicts(self):
        keywords_rep = {'ampx200b': "", 'x200b': "", 'AITA': "", 'aita': ""}
        
        for key in NTA_KEYWORDS + YTA_KEYWORDS:
            keywords_rep[key] = ""
        keywords_rep = dict(sorted(keywords_rep.items(), key=lambda k: len(k[0]), reverse=True))

        rep = dict((re.escape(k), v) for k, v in keywords_rep.items()) 
        pattern = re.compile("|".join(rep.keys()))

        for verdict, text in tqdm(self.verdictToText.items(), desc="Removing keywords from verdicts"):
            self.verdictToCleanedText[verdict] = pattern.sub(lambda m: rep[re.escape(m.group(0))], text.lower())
            self.verdictToCleanedText[verdict] = self.verdictToCleanedText[verdict].translate(str.maketrans('', '', string.punctuation))       
    
    def clean_single_text(self, text):
        keywords_rep = {'ampx200b': "", 'x200b': "", 'AITA': "", 'aita': "", '[removed]': "", '[REMOVED]': "", '[deleted]': "", '[DELETED]': ""}
        rep = dict((re.escape(k), v) for k, v in keywords_rep.items()) 
        pattern = re.compile("|".join(rep.keys()))
        text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text.lower())
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def clean_single_verdict(self, text):
        keywords_rep = {'ampx200b': "", 'x200b': "", 'AITA': "", 'aita': "", '[removed]': "", '[REMOVED]': "", '[deleted]': "", '[DELETED]': ""}
        for key in NTA_KEYWORDS + YTA_KEYWORDS:
            keywords_rep[key] = ""
        keywords_rep = dict(sorted(keywords_rep.items(), key=lambda k: len(k[0]), reverse=True))
        
        rep = dict((re.escape(k), v) for k, v in keywords_rep.items()) 
        pattern = re.compile("|".join(rep.keys()))
        text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text.lower())
        return text.translate(str.maketrans('', '', string.punctuation))    
    
    
    def get_authors_from_situations(self, situations):
        authors = set()
        for sit in situations:
            if sit in self.postToVerdicts:
                verdicts = self.postToVerdicts[sit]
                for v in verdicts:
                    authors.add(self.verdictToAuthor[v])
        
        return authors
    
            
class VerdictDataset:
    """Creates the verdicts datastructures to use through out the project
    """
    def __init__(self, authors_vocab):
        self.authors_otherComments, self.authors_Verdicts = split_verdicts_comments_amit(authors_vocab)
        self.verdictToId = dict()
        self.idToVerdict = list()
        self.verdictToText = dict()
        self.verdictToCleanedText = dict()
        self.verdictToParent = dict()
        
        self.verdictToLabel = dict()
        
        self.authorToVerdict = ListDict()
        self.verdictToAuthor = dict()
        self.author_labeledVerdicts = ListDict()
        self.load_maps_and_labels()
        self._clean_keywords_from_verdicts()
    
    
    def load_maps_and_labels(self):
        """Iterate through authors vocabulary, and fill the dictionaries
        

        Returns:
            _type_: _description_
        """
        self.author_labeledVerdicts = ListDict()

        for author, verdicts_id in tqdm(self.authors_Verdicts.items(), desc="Creating dictionary maps for the dataset"):
            for text, verdictId, parent_id in verdicts_id:
                self.verdictToText[verdictId] = text
                self.verdictToParent[verdictId] = parent_id
                self.authorToVerdict.append(author, verdictId)
                self.verdictToAuthor[verdictId] = author
                
                if verdictId not in self.verdictToId:
                    self.idToVerdict.append(verdictId)
                    self.verdictToId[verdictId] = len(self.idToVerdict) - 1
                
                label = self.get_comment_label(text)
                if label != None:
                    self.verdictToLabel[verdictId] = label
                    self.author_labeledVerdicts.append(author, (verdictId, label))
                        
        return self.author_labeledVerdicts

    
    def get_comment_label(self, text):
        text = remove_extra_spaces(text)
    
        for key in YTA_KEYWORDS:
            if key in text.lower():
                return 1
        for key in NTA_KEYWORDS:
            if key in text.lower():
                return 0
        
        return None
    
        
    def _clean_keywords_from_verdicts(self):
        keywords_rep = {'ampx200b': "", 'x200b': ""}
        for key in NTA_KEYWORDS + YTA_KEYWORDS:
            keywords_rep[key] = ""
        keywords_rep = dict(sorted(keywords_rep.items(), key=lambda k: len(k[0]), reverse=True))

        rep = dict((re.escape(k), v) for k, v in keywords_rep.items()) 
        pattern = re.compile("|".join(rep.keys()))

        for verdict, text in tqdm(self.verdictToText.items(), desc="Removing keywords from verdicts"):
            self.verdictToCleanedText[verdict] = pattern.sub(lambda m: rep[re.escape(m.group(0))], text.lower())
            self.verdictToCleanedText[verdict] = self.verdictToCleanedText[verdict].translate(str.maketrans('', '', string.punctuation))

