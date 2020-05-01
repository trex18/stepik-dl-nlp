import collections
import re
import scipy.sparse
import numpy as np
import nltk
from nltk import ngrams
from nltk.corpus import wordnet


TOKEN_RE = re.compile(r'[\w\d]+')


def tokenize_text_simple_regex(txt, min_token_size=4):
    txt = txt.lower()
    all_tokens = TOKEN_RE.findall(txt)
    return [token for token in all_tokens if len(token) >= min_token_size]


def character_tokenize(txt):
    return list(txt)


def tokenize_corpus(texts, tokenizer=tokenize_text_simple_regex, 
                    stemmer=None, lemmatizer=None, **tokenizer_kwargs):
    """
    A function to tokenize a corpus of texts
    Can lemmatize or stem the tokens (not both at the same time)
    """
    corpus_tokens = []
    if stemmer:
        corpus_tokens = [
            [
                stemmer.stem(token) for token in tokenizer(text, **tokenizer_kwargs)
            ] for text in texts
        ]
            
    elif lemmatizer:        
        corpus_tokens = [
            [
                lemmatizer.lemmatize(token, get_wordnet_pos(token)) \
                    for token in tokenizer(text, **tokenizer_kwargs)
            ] for text in texts
        ]
    else:
        corpus_tokens = [tokenizer(text, **tokenizer_kwargs) for text in texts]
    
    return corpus_tokens

def ngramize_corpus(texts, n, all_up_to_n=False, 
                    tokenizer=tokenize_text_simple_regex, stemmer=None,
                    lemmatizer=None, **tokenizer_kwargs):
    assert n >= 1
    
    corpus_ngrams = []
    for text in texts:
        text_ngrams = []
        if stemmer:
            tokenized_text =  [
                stemmer.stem(token) for token in tokenizer(text, **tokenizer_kwargs)
            ]
        elif lemmatizer:
            tokenized_text =  [
                lemmatizer.lemmatize(token, get_wordnet_pos(token)) \
                    for token in tokenizer(text, **tokenizer_kwargs)
            ]
        else:
            tokenized_text =  tokenizer(text, **tokenizer_kwargs)
            
        if all_up_to_n:
            for k in range(1, n + 1):
                    for gram in ngrams(tokenized_text, k):
                        text_ngrams.append(gram)
        else:
            text_ngrams = [gram for gram in ngrams(tokenizer(text, **tokenizer_kwargs), n)]
            
        corpus_ngrams.append(text_ngrams)
    return corpus_ngrams
        
def add_fake_token(word2id, token='<PAD>'):
    word2id_new = {token: i + 1 for token, i in word2id.items()}
    word2id_new[token] = 0
    return word2id_new


def texts_to_token_ids(tokenized_texts, word2id):
    return [[word2id[token] for token in text if token in word2id]
            for text in tokenized_texts]


def build_vocabulary(tokenized_texts, max_size=1000000, max_doc_freq=0.8, min_count=5, pad_word=None):
    word_counts = collections.defaultdict(int)
    doc_n = 0

    # посчитать количество документов, в которых употребляется каждое слово
    # а также общее количество документов
    for txt in tokenized_texts:
        doc_n += 1
        unique_text_tokens = set(txt)
        for token in unique_text_tokens:
            word_counts[token] += 1

    # убрать слишком редкие и слишком частые слова
    word_counts = {word: cnt for word, cnt in word_counts.items()
                   if cnt >= min_count and cnt / doc_n <= max_doc_freq}

    # отсортировать слова по убыванию частоты
    sorted_word_counts = sorted(word_counts.items(),
                                reverse=True,
                                key=lambda pair: pair[1])

    # добавим несуществующее слово с индексом 0 для удобства пакетной обработки
    if pad_word is not None:
        sorted_word_counts = [(pad_word, 0)] + sorted_word_counts

    # если у нас по прежнему слишком много слов, оставить только max_size самых частотных
    if len(word_counts) > max_size:
        sorted_word_counts = sorted_word_counts[:max_size]

    # нумеруем слова
    word2id = {word: i for i, (word, _) in enumerate(sorted_word_counts)}

    # нормируем частоты слов
    word2freq = np.array([cnt / doc_n for _, cnt in sorted_word_counts], dtype='float32')

    return word2id, word2freq

def document_word_occurences(tokenized_texts, word2id, labels, n_classes):
    """
    this function returns a matrix A (n_classes x n_tokens): 
    A[L, w] = word w is contained in A[L, w] documents of class L
    """
    result = scipy.sparse.dok_matrix((n_classes, len(word2id)), dtype=np.float32)
    for text_id, text in enumerate(tokenized_texts):
        unique_tokens = set(text)
        for token in unique_tokens:
            if token in word2id.keys():
                result[labels[text_id], word2id[token]] += 1
    return result

def get_wordnet_pos(word):
    """
    Map POS tag to first character lemmatize() accepts
    This function is used in lemmatizers
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

PAD_TOKEN = '__PAD__'
NUMERIC_TOKEN = '__NUMBER__'
NUMERIC_RE = re.compile(r'^([0-9.,e+\-]+|[mcxvi]+)$', re.I)


def replace_number_nokens(tokenized_texts):
    return [[token if not NUMERIC_RE.match(token) else NUMERIC_TOKEN for token in text]
            for text in tokenized_texts]
