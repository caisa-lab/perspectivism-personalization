import random 
from tqdm import tqdm
import numpy as np

def sampled_print(iter_obj, nr_values=10):
    if type(iter_obj) == set:
        print(random.sample(list(iter_obj), nr_values))
    else:
        print(random.sample(iter_obj, nr_values))
        
def get_top_ind_per_cluster(count_vectorizer, cluster_text, top_k=10, n_clusters=20):
    top_indeces = []
    counts = []
    
    for cluster in tqdm(range(n_clusters)):
        counts_cluster = count_vectorizer.transform(cluster_text[cluster]).toarray()
        temp_c = counts_cluster.sum(axis=0)
        counts.append(temp_c)
        top_indeces.append(np.argpartition(temp_c, -top_k)[-top_k:])
    
    return top_indeces, counts

def get_top_ngrams(count_vectorizer, top_indeces):
    feature_names = count_vectorizer.get_feature_names_out()
    top_ngram_cluster = []

    for top in top_indeces:
        temp = []
        for ind in top:
            temp.append(feature_names[ind])

        top_ngram_cluster.append(temp)
    
    return top_ngram_cluster


class IterDict(dict):
    def __init__(self):
        super(IterDict, self)
    
    def reverse_dict(self, n_values=20):
        reverse_dict = {}

        for i in range(n_values):    
            reverse_dict[i] = set()

        for key, value in self.items(): 
            if type(value) == list or type(value) == set:
                for v in value:
                    reverse_dict[v].add(key)
            else:
                reverse_dict[value].add(key)
                
        return reverse_dict
    

class ListDict(IterDict):
    def __init__(self):
        super(ListDict, self)
        
    def append(self, key, value):
        current_values = self.get(key, [])
        current_values.append(value)
        self[key] = current_values
    
    def update_lists(self, other):
        for key, value in other.items():
            curr_v = self.get(key, [])
            curr_v.extend(value)
            self[key] = curr_v
        
class SetDict(IterDict):
    def __init__(self):
        super(SetDict, self)
        
    def add(self, key, value):
        current_values = self.get(key, set())
        current_values.add(value)
        self[key] = current_values
    