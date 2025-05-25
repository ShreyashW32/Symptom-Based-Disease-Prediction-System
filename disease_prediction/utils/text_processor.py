import numpy as np
import string


def tokenize(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    return tokens


def build_vocabulary(train_texts):
    all_words = set()
    for text in train_texts:
        tokens = tokenize(text)
        all_words.update(tokens)
    vocabulary = list(all_words)
    word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
    return vocabulary, word_to_idx


def texts_to_vectors(texts, word_to_idx):
    vectors = []
    for text in texts:
        tokens = tokenize(text)
        vector = np.zeros(len(word_to_idx))
        for token in tokens:
            if token in word_to_idx:
                vector[word_to_idx[token]] += 1
        vectors.append(vector)
    return np.array(vectors)
