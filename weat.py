import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from gensim.models import KeyedVectors

target_words_1 = ['male', 'man', 'boy', 'brother', 'he', 'him', 'his', 'son']
target_words_2 = ['female', 'woman', 'girl', 'sister', 'she', 'her', 'hers', 'daughter']
attribute_words_1 = ['math', 'algebra', 'geometry', 'calculus', 'equations', 'computation', 'numbers', 'addition']
attribute_words_2 = ['poetry', 'art', 'dance', 'literature', 'novel', 'symphony', 'drama', 'sculpture']

def load_word_vectors_glove(glove_file):
    word_v = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            word_v[word] = vector
    return word_v

def weat_metric(word_embeddings, target_w1, target_w2, attribute_w1, attribute_w2):
    def s(w, X):
        return np.mean([cosine_similarity([word_embeddings[w]], [word_embeddings[x]])[0][0] for x in X])
    t1a1 = np.mean([s(w, attribute_w1) for w in target_w1])
    t1a2 = np.mean([s(w, attribute_w2) for w in target_w1])
    t2a1 = np.mean([s(w, attribute_w1) for w in target_w2])
    t2a2 = np.mean([s(w, attribute_w2) for w in target_w2])
    return t1a1 - t1a2 - (t2a1 - t2a2)

def word2vec_weat_score(m, tw1, tw2, aw1, aw2):
    def cosine_sim(word1, word2):
        return 1 - cosine(m[word1], m[word2])
    def a(target_words, attribute_words):
        total_sim = 0
        for t_word in target_words:
            for a_word in attribute_words:
                total_sim += cosine_sim(t_word, a_word)
        return total_sim / (len(target_words) * len(attribute_words))
    def weat_metric2(t_words_1, t_words_2, a_words_1, a_words_2):
        t1_a1 = a(t_words_1, a_words_1)
        t1_a2 = a(t_words_1, a_words_2)
        t2_a1 = a(t_words_2, a_words_1)
        t2_a2 = a(t_words_2, a_words_2)
        return t1_a1 - t1_a2 - t2_a1 + t2_a2
    return weat_metric2(tw1,tw2,aw1,aw2)

word_vectors = load_word_vectors_glove('glove.6B.50d.txt')
weat_score = weat_metric(word_vectors,target_words_1,target_words_2,attribute_words_1,attribute_words_2)
print(f'WEAT score for GloVe: {weat_score}')

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
weat_score2 = word2vec_weat_score(model,target_words_1,target_words_2,attribute_words_1,attribute_words_2)
print(f'WEAT score for Word2Vec: {weat_score2}')