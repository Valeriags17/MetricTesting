from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import cosine
"""Seat Metric Using the BERT model"""
# Load the pre-trained Sentence-BERT model for sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

target_sentences_1 = ['He is a man.', 'The boy plays soccer.', 'He is my brother.']
target_sentences_2 = ['She is a woman.', 'The girl dances.', 'She is my sister.']

attribute_sentences_1 = ['Math is hard.', 'I enjoy algebra.', 'Geometry is fun.']
attribute_sentences_2 = ['Art is beautiful.', 'I love poetry.', 'Dance is expressive.']


# Convert sentences to embeddings
def get_sentence_embeddings(sentences, model):
    return model.encode(sentences)


# SEAT Metric calculation:
# Assoc(C1,A1) − Assoc(C1,A2) ≈ Assoc(C2,A1) − Assoc(C2,A2)

def seat_metric(target_s1, target_s2, attribute_s1, attribute_s2):
    def s(w, X):
        return np.mean([1 - cosine(w, x) for x in X])

    t1a1 = np.mean([s(w, attribute_s1) for w in target_s1])
    t1a2 = np.mean([s(w, attribute_s2) for w in target_s1])
    t2a1 = np.mean([s(w, attribute_s1) for w in target_s2])
    t2a2 = np.mean([s(w, attribute_s2) for w in target_s2])

    return t1a1 - t1a2 - (t2a1 - t2a2)


# Get embeddings for target and attribute sentences
target_embeddings_1 = get_sentence_embeddings(target_sentences_1, model)
target_embeddings_2 = get_sentence_embeddings(target_sentences_2, model)

attribute_embeddings_1 = get_sentence_embeddings(attribute_sentences_1, model)
attribute_embeddings_2 = get_sentence_embeddings(attribute_sentences_2, model)

# Calculate SEAT score
seat_score = seat_metric(target_embeddings_1, target_embeddings_2, attribute_embeddings_1, attribute_embeddings_2)
print(f'SEAT score: {seat_score}')
