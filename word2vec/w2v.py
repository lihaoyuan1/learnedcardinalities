from gensim.models import KeyedVectors


class Word2Vec:

    def __init__(self, path):
        self.wv = KeyedVectors.load(path, mmap='r')

    def word_to_vector(self, word):
        return self.wv[word]
