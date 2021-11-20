import networkx as nx
import numpy as np
from numpy import linalg as la
from sklearn.preprocessing import normalize

from models.base_model import BaseModel


class Node2Vec_file(BaseModel):
    def __init__(self, graph, features, labels=None, dim=80):
        super(Node2Vec_file, self).__init__(graph, features, dim, labels)

    def learn_embeddings(self):

        file = open('/Users/makarov/Downloads/node2vec-master/emb/node2vec_emb.txt','r')

        embs = {}
        i = 0
        for line in file:
            if i != 0:
                lines = line.split()
                node = int(lines[0])
                emb =  [ float(number) for number in lines[1:]]
                embs[node] = emb
            i +=1
        file.close() 
        
        
        embeddings = []
        for node in self.graph.nodes():
            embeddings.append(embs[node])
        
        self.embeddings = np.array(embeddings)