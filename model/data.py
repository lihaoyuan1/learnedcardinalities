import csv
import logging

import numpy as np
import torch
from torch.utils.data import dataset

from utils import get_all_binding_names, get_set_encoding, get_all_predicate_names, get_all_joins, encode_samples, \
    encode_data, normalize_labels

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger('data')


def load_data(file_name, num_materialized_samples):
    predicates = []
    joins = []
    bindings = []
    samples = []
    label = []

    with open(file_name + '.csv', 'rU') as f:
        data_row = list(list(rec) for rec in csv.reader(f))
        for row in data_row:
            predicates.append(row[0].split('|'))
            # 去除空值
            joins.append(row[1].split('|'))
            # 去除空值
            bindings.append(row[2].split('|'))
            if int(row[3]) < 1:
                logger.error('Queries must have non-zero cardinalities')
                exit(1)
            label.append(row[3])
    logger.info('Loaded queries')

    # Load bitmaps
    num_bytes_per_bitmap = int((num_materialized_samples + 7) >> 3)
    with open(file_name + ".bitmaps", 'rb') as f:
        for i in range(len(predicates)):
            four_bytes = f.read(4)
            if not four_bytes:
                print("Error while reading 'four_bytes'")
                exit(1)
            num_bitmaps_curr_query = int.from_bytes(four_bytes, byteorder='little')
            bitmaps = np.empty((num_bitmaps_curr_query, num_bytes_per_bitmap * 8), dtype=np.uint8)
            for j in range(num_bitmaps_curr_query):
                # Read bitmap
                bitmap_bytes = f.read(num_bytes_per_bitmap)
                if not bitmap_bytes:
                    logger.error("Error while reading 'bitmap_bytes'")
                    exit(1)
                bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
            samples.append(bitmaps)
    logger.info('Loaded bitmaps')

    # Split bindings
    bindings_ = []
    for binding in bindings:
        bindings_.append([b.split('=') for b in binding])

    return predicates, joins, bindings_, samples, label


def load_and_encode_train_data(num_queries, num_materialized_samples):
    file_name_queries = 'data/train'

    predicates, joins, bindings, samples, label = load_data(file_name_queries, num_materialized_samples)

    # Get binding name dict
    binding_names = get_all_binding_names(bindings)
    binding2vec, idx2binding = get_set_encoding(binding_names)

    # Get predicate name dict
    predicate_names = get_all_predicate_names(predicates)
    predicate2vec, idx2predicate = get_set_encoding(predicate_names)

    # Get join name dict
    join_set = get_all_joins(joins)
    join2vec, idx2join = get_set_encoding(join_set)

    # Get feature encoding and proper normalization
    samples_enc = encode_samples(predicates, samples, predicate2vec)
    bindings_enc, joins_enc = encode_data(bindings, joins, binding2vec, join2vec)
    label_norm, min_val, max_val = normalize_labels(label)

    # Split in training and validation samples
    num_train = int(num_queries * 0.9)
    num_test = num_queries - num_train

    samples_train = samples_enc[:num_train]
    bindings_train = bindings_enc[:num_train]
    joins_train = joins_enc[:num_train]
    labels_train = label_norm[:num_train]

    samples_test = samples_enc[num_train:num_train + num_test]
    bindings_test = bindings_enc[num_train:num_train + num_test]
    joins_test = joins_enc[num_train:num_train + num_test]
    labels_test = label_norm[num_train:num_train + num_test]

    logger.info('Number of training samples: {}'.format(len(labels_train)))
    logger.info('Number of validation samples: {}'.format(len(labels_test)))

    max_num_joins = max(max([len(j) for j in joins_train]), max([len(j) for j in joins_test]))
    max_num_bindings = max(max([len(p) for p in bindings_train]), max([len(p) for p in bindings_test]))

    dicts = [predicate2vec, binding2vec, join2vec]
    train_data = [samples_train, bindings_train, joins_train]
    test_data = [samples_test, bindings_test, joins_test]

    return dicts, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_bindings, train_data, test_data


def make_dataset(samples, bindings, joins, labels, max_num_joins, max_num_bindings):
    """Add zero-padding and wrap as tensor dataset."""

    sample_masks = []
    sample_tensors = []
    for sample in samples:
        sample_tensor = np.vstack(sample)
        num_pad = max_num_joins + 1 - sample_tensor.shape[0]
        sample_mask = np.ones_like(sample_tensor).mean(1, keepdims=True)
        sample_tensor = np.pad(sample_tensor, ((0, num_pad), (0, 0)), 'constant')
        sample_mask = np.pad(sample_mask, ((0, num_pad), (0, 0)), 'constant')
        sample_tensors.append(np.expand_dims(sample_tensor, 0))
        sample_masks.append(np.expand_dims(sample_mask, 0))
    sample_tensors = np.vstack(sample_tensors)
    sample_tensors = torch.FloatTensor(sample_tensors)
    sample_masks = np.vstack(sample_masks)
    sample_masks = torch.FloatTensor(sample_masks)

    binding_masks = []
    binding_tensors = []
    for binding in bindings:
        binding_tensor = np.vstack(binding)
        num_pad = max_num_bindings - binding_tensor.shape[0]
        binding_mask = np.ones_like(binding_tensor).mean(1, keepdims=True)
        binding_tensor = np.pad(binding_tensor, ((0, num_pad), (0, 0)), 'constant')
        binding_mask = np.pad(binding_mask, ((0, num_pad), (0, 0)), 'constant')
        binding_tensors.append(np.expand_dims(binding_tensor, 0))
        binding_masks.append(np.expand_dims(binding_mask, 0))
    binding_tensors = np.vstack(binding_tensors)
    binding_tensors = torch.FloatTensor(binding_tensors)
    binding_masks = np.vstack(binding_masks)
    binding_masks = torch.FloatTensor(binding_masks)

    join_masks = []
    join_tensors = []
    for join in joins:
        join_tensor = np.vstack(join)
        num_pad = max_num_joins - join_tensor.shape[0]
        join_mask = np.ones_like(join_tensor).mean(1, keepdims=True)
        join_tensor = np.pad(join_tensor, ((0, num_pad), (0, 0)), 'constant')
        join_mask = np.pad(join_mask, ((0, num_pad), (0, 0)), 'constant')
        join_tensors.append(np.expand_dims(join_tensor, 0))
        join_masks.append(np.expand_dims(join_mask, 0))
    join_tensors = np.vstack(join_tensors)
    join_tensors = torch.FloatTensor(join_tensors)
    join_masks = np.vstack(join_masks)
    join_masks = torch.FloatTensor(join_masks)

    target_tensor = torch.FloatTensor(labels)

    return dataset.TensorDataset(sample_tensors, binding_tensors, join_tensors, target_tensor, sample_masks,
                                 binding_masks, join_masks)


def get_train_datasets(num_queries, num_materialized_samples):
    dicts, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_bindings, train_data, test_data = \
        load_and_encode_train_data(num_queries, num_materialized_samples)
    train_dataset = make_dataset(*train_data, labels=labels_train, max_num_joins=max_num_joins,
                                 max_num_bindings=max_num_bindings)
    logger.info('Created TensorDataset for training data')
    test_dataset = make_dataset(*test_data, labels=labels_test, max_num_joins=max_num_joins,
                                max_num_bindings=max_num_bindings)
    logger.info('Created TensorDataset for validation data')
    return dicts, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_bindings, train_dataset, test_dataset
