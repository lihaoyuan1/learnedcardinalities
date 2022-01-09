import csv
import json
import random
import struct

import numpy as np

from endpoint import Endpoint


def load_predicates(file_name):
    predicates = list()
    with open(file_name, 'r') as f:
        rows = csv.reader(f)
        for row in rows:
            predicates.append(row)

    return predicates


def load_sample(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)


def sample(size):
    repository = Endpoint('http://182.61.55.66:7200/repositories/lubm')

    predicates = load_predicates('../data/predicate.csv')

    sample_map = {}

    for predicate, cardinality in predicates:
        if int(cardinality) <= size:
            samples = repository.query_with_predicate(predicate)
        else:
            samples = list()
            offset_set = set()
            while len(samples) != size:
                offset = random.randint(0, int(cardinality) - 1)
                if offset in offset_set:
                    continue
                s, o = repository.random_query_with_predicate(predicate, offset)
                samples.append([s, o])
        sample_map[predicate] = samples

    with open('../data/sample.json', 'w+') as f:
        json.dump(sample_map, f)


def generate_bitmaps():
    predicates = load_predicates('../data/predicate.csv')
    samples = load_sample('../data/sample.json')

    predicate_map = {}
    for predicate, _ in predicates:
        predicate_map[predicate.split('#')[1]] = predicate

    with open('../workload/join0/join0.csv', 'r') as f1, open('../workload/join0/joino.bitmaps', 'wb') as f2:
        rows = csv.reader(f1)

        for row in rows:
            predicate_names = list(row[0].split('|'))
            bindings = list(row[2].split('|'))

            binding_map = {}
            for binding in bindings:
                if len(binding) == 0:
                    continue
                b = binding.split('=')
                binding_map[b[0]] = b[1]

            f2.write(struct.pack('i', len(predicate_names)))

            for name in predicate_names:
                predicate = predicate_map.get(name)
                sample_list = samples[predicate]

                bitmaps = []
                for i in range(1000):
                    if i >= len(sample_list):
                        bitmaps.append(0)
                        continue
                    s, o = sample_list[i]
                    if binding_map.get(name + '.s') is not None:
                        bitmaps.append(1 if s == binding_map.get(name + '.s') else 0)
                    elif binding_map.get(name + '.o') is not None:
                        bitmaps.append(1 if o == binding_map.get(name + '.o') else 0)
                    else:
                        bitmaps.append(1)
                f2.write(bytes(np.packbits(bitmaps)))


if __name__ == '__main__':
    # sample(1000)
    generate_bitmaps()
