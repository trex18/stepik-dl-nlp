import numpy as np
import scipy.sparse
import torch
from torch.utils.data import Dataset
from collections import Counter

def pmi(classes_x_tokens, labels):
    """
    returns two matrices, pmi(l, w) and pmi(l, neg(w)) of shape n_classes x n_tokens
    """
    classes_counter = Counter(labels)
    classes_counts = np.array([classes_counter[i] for i in sorted(classes_counter.keys())])

    # these are the probabilities of each class
    p_l = classes_counts / classes_counts.sum() 

    # p_w are the probabilities of each word
    p_w = classes_x_tokens.sum(0) / classes_counts.sum()
    # this is the probability that the word w doesn't occur
    p_w2 = 1.0 - p_w
    

    #p_lw is the probability of a word w occuring in a document of a class l
    p_lw = np.multiply(classes_x_tokens, 1.0 / np.expand_dims(classes_counts, 1))
    p_lw2 = 1.0 - p_lw

    
    pmi = np.multiply(p_lw, 1.0 / np.expand_dims(p_l, 1))
    pmi = np.multiply(pmi, 1.0 / p_w)
    
    pmi2 = np.multiply(p_lw2, 1.0 / np.expand_dims(p_l, 1))
    pmi2 = np.multiply(pmi2, 1.0 / p_w2)
    
    return pmi, pmi2

def informativeness(pmi, pmi_neg):
    """
    Given matrices of pointwise mutual information of shape n_classes x n_tokens
    returns an array of informativeness for each word
    informativeness = min_l[pmi(l, w)] + min_l[pmi(l, neg_w)]
    """
    return np.ravel(pmi.max(axis=0) + pmi_neg.max(axis=0))

def vectorize_texts(tokenized_texts, word2id, word2freq, 
                    mode='tfidf', scale=True, info_vector=None):
    assert mode in {'tfidf', 'idf', 'tf', 'bin', 'pmi'}

    # считаем количество употреблений каждого слова в каждом документе
    result = scipy.sparse.dok_matrix((len(tokenized_texts), len(word2id)), dtype='float32')
    for text_i, text in enumerate(tokenized_texts):
        for token in text:
            if token in word2id:
                result[text_i, word2id[token]] += 1

    # получаем бинарные вектора "встречается или нет"
    if mode == 'bin':
        result = (result > 0).astype('float32')

    # получаем вектора относительных частот слова в документе
    elif mode == 'tf':
        result = result.tocsr()
        result = result.multiply(1 / result.sum(1))

    # полностью убираем информацию о количестве употреблений слова в данном документе,
    # но оставляем информацию о частотности слова в корпусе в целом
    elif mode == 'idf':
        result = (result > 0).astype('float32').multiply(1 / word2freq)

    # учитываем всю информацию, которая у нас есть:
    # частоту слова в документе и частоту слова в корпусе
    elif mode == 'tfidf':
        result = result.tocsr()
        result = result.multiply(1 / result.sum(1))  # разделить каждую строку на её длину
        result = result.multiply(1 / word2freq)  # разделить каждый столбец на вес слова
        
    elif mode == 'pmi':
        result = result.tocsc()    
        result = result.multiply(info_vector)

    if scale:
        result = result.tocsc()
        result -= result.min()
        result /= (result.max() + 1e-6)

    return result.tocsr()


class SparseFeaturesDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        cur_features = torch.from_numpy(self.features[idx].toarray()[0]).float()
        cur_label = torch.from_numpy(np.asarray(self.targets[idx])).long()
        return cur_features, cur_label
