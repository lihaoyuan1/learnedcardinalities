import csv

from gensim.models import Word2Vec, KeyedVectors


def train_model():
    with open('../data/lubm100.csv', 'r') as f:
        sentences = list(list(rec) for rec in csv.reader(f))

    print(len(sentences))

    model = Word2Vec(sentences, min_count=1, sg=1)

    model.wv.save('w2v.wordvectors')

    print(len(model.wv.index_to_key))


if __name__ == '__main__':
    train_model()
