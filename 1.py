import numpy as np
from numpy.random import dirichlet
from numpy.random import choice as np_choice
from numpy.linalg import norm 
import json
from sklearn.decomposition import LatentDirichletAllocation as LDA

def create_phi_matrix(K,V):
    num_words_per_topic = V // K
    phi_matrix = np.zeros((K,V))

    high_prob_share = 0.90
    low_prob_share = 0.10

    HIGH_PROB = high_prob_share /num_words_per_topic
    LOW_PROB = low_prob_share / (V - num_words_per_topic)
