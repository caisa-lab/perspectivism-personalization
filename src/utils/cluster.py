from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from .clusters_utils import *

class Cluster:
    def __init__(self, type, n_clusters, rot_texts, rot_id, situations, situation_id, sitids_content, random_state):
        self.type = type
        self.n_clusters = n_clusters
        self.kmModel = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.rot_texts = rot_texts
        self.rot_id = rot_id
        self.situations = situations
        self.situation_id = situation_id
        self.sitids_content = sitids_content
        assert len(situations) == len(rot_texts)
        assert len(rot_texts) == len(rot_id)
        assert len(situations) == len(situation_id)
        
    def fit(self, vectorizer):
        if self.type == 'rot':
            print("Fitting clusters of type {}".format(self.type))
            features = vectorizer.fit_transform(self.rot_texts)
            tf_idf_norm = normalize(features)
            tf_idf_array = tf_idf_norm.toarray()
        elif self.type == 'situation':
            print("Fitting clusters of type {}".format(self.type))
            features = vectorizer.fit_transform(self.situations)
            tf_idf_norm = normalize(features)
            tf_idf_array = tf_idf_norm.toarray()
        else:
            raise Exception("Cluster is initialized with wrong type {}".format(self.type))
        self.kmModel.fit(tf_idf_array)
    
    def create_maps(self):
        print("Mapping ROT clusters")
        self.rot_labels = dict(zip(self.rot_texts, self.kmModel.labels_))
        self.situationsId_labels = dict(zip(self.situation_id, self.kmModel.labels_))
        self.cluster_rot = {}

        for i in range(self.n_clusters):
            self.cluster_rot[i] = []
            
        for rot, cluster in self.rot_labels.items():
            self.cluster_rot[cluster].append(rot)
            
        print("Mapping Situation to Clusters")
        # Extract situations labels, based on ROT clustering
        # Take care to place one situation in multiple clusters

        situations_labels = ListDict()
        sitId_labels = SetDict()

        for idx, label in enumerate(self.kmModel.labels_):
            situation = self.situations[idx]
            sit_id = self.situation_id[idx]
            
            situations_labels.append(situation, label)
            sitId_labels.add(sit_id, label)

        self.cluster_situation = situations_labels.reverse_dict(n_values=self.n_clusters)
        self.cluster_sitId = sitId_labels.reverse_dict(n_values=self.n_clusters)
        
        self.cluster_sitContent = {}
        for cluster, sitIds in self.cluster_sitId.items():
            self.cluster_sitContent[cluster] = [self.sitids_content[sitId] for sitId in sitIds if sitId in self.sitids_content]
        
            
        
        
        
        
            
        
        
