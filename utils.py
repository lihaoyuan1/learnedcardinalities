import numpy as np

from word2vec.w2v import Word2Vec


def normalization(data):
    s = sum(data)
    return data / s


def standard_format(value):
    if value.startswith('?'):
        return value
    elif value.startswith('http'):
        return '<' + value + '>'
    else:
        return '\'' + value + '\''


def get_all_binding_names(bindings):
    binding_names = set()
    for binding in bindings:
        for b in binding:
            if len(b) == 2:
                binding_names.add(b[0])

    return binding_names


def get_all_predicate_names(predicates):
    predicate_names = set()
    for query in predicates:
        for predicate in query:
            predicate_names.add(predicate)

    return predicate_names


def get_all_joins(joins):
    join_set = set()
    for query in joins:
        for join in query:
            join_set.add(join)
    return join_set


def idx_to_one_hot(idx, num_elements):
    one_hot = np.zeros(num_elements, dtype=np.float32)
    one_hot[idx] = 1.
    return one_hot


def get_set_encoding(source_set, one_hot=True):
    num_elements = len(source_set)
    source_list = list(source_set)
    # Sort list to avoid non-deterministic behavior
    source_list.sort()
    # Build map from s to i
    thing2idx = {s: i for i, s in enumerate(source_list)}
    # Build array (essentially a map from idx to s)
    idx2thing = [s for i, s in enumerate(source_list)]
    if one_hot:
        thing2vec = {s: idx_to_one_hot(i, num_elements) for i, s in enumerate(source_list)}
        return thing2vec, idx2thing
    return thing2idx, idx2thing


def encode_samples(predicates, samples, predicate2vec):
    samples_enc = []
    for i, query in enumerate(predicates):
        samples_enc.append(list())
        for j, predicate in enumerate(query):
            sample_vec = []
            # Append table one-hot vector
            sample_vec.append(predicate2vec[predicate])
            # Append bit vector
            sample_vec.append(samples[i][j])
            sample_vec = np.hstack(sample_vec)
            samples_enc[i].append(sample_vec)

    return samples_enc


def encode_data(bindings, joins, binding2vec, join2vec):
    bindings_enc = []
    joins_enc = []
    word2vec = Word2Vec('word2vec/w2v.wordvectors')

    for i, query in enumerate(bindings):
        bindings_enc.append(list())
        joins_enc.append(list())
        for binding in query:
            if len(binding) == 2:
                # Proper predicate
                bind = binding[0]
                val = binding[1]
                vec_val = word2vec.word_to_vector(val)

                bind_vec = []
                bind_vec.append(binding2vec[bind])
                bind_vec.append(vec_val)
                bind_vec = np.hstack(bind_vec)
            else:
                bind_vec = np.zeros((len(binding2vec) + 100))

            bindings_enc[i].append(bind_vec)

        for join in joins[i]:
            # Join instruction
            join_enc = join2vec[join]
            joins_enc[i].append(join_enc)

    return bindings_enc, joins_enc


def normalize_labels(labels, min_val=None, max_val=None):
    labels = np.array([np.log(float(label)) for label in labels])

    if min_val is None:
        min_val = labels.min()
        print("min log(label): {}".format(min_val))
    if max_val is None:
        max_val = labels.max()
        print("max log(label): {}".format(max_val))
    labels_norm = (labels - min_val) / (max_val - min_val)
    # Threshold labels
    labels_norm = np.minimum(labels_norm, 1)
    labels_norm = np.maximum(labels_norm, 0)
    return labels_norm, min_val, max_val


def unnormalize_labels(labels_norm, min_val, max_val):
    labels_norm = np.array(labels_norm, dtype=np.float32)
    labels = (labels_norm * (max_val - min_val)) + min_val
    return np.array(np.round(np.exp(labels)), dtype=np.int64)
