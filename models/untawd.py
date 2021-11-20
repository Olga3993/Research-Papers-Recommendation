import numpy as np
import pandas as pd
import os
import shutil

from pathlib import Path
from datetime import datetime
from subprocess import run

from models.base_model import BaseModel

class UnTADW(BaseModel):
    def __init__(self, graph, features, labels=None, dim=128):
        super(UnTADW, self).__init__(graph, features, dim, labels)


    def learn_embeddings(self):

        file = open('/Users/makarov/Downloads/OpenANE-master/emb/w300_abrw_emb.txt','r')

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


