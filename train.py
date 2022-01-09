import logging
import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model.data import get_train_datasets, load_data, make_dataset
from model.model import SetConv
from utils import unnormalize_labels, encode_samples, encode_data, normalize_labels

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger('train')


def unnormalize_torch(vals, min_val, max_val):
    vals = (vals * (max_val - min_val)) + min_val
    return torch.exp(vals)


def qerror_loss(preds, targets, min_val, max_val):
    qerror = []
    preds = unnormalize_torch(preds, min_val, max_val)
    targets = unnormalize_torch(targets, min_val, max_val)

    for i in range(len(targets)):
        if (preds[i] > targets[i]).cpu().data.numpy()[0]:
            qerror.append(preds[i] / targets[i])
        else:
            qerror.append(targets[i] / preds[i])
    return torch.mean(torch.cat(qerror))


def predict(model, data_loader, cuda):
    preds = []
    t_total = 0.

    model.eval()
    for batch_idx, data_batch in enumerate(data_loader):

        samples, bindings, joins, targets, sample_masks, binding_masks, join_masks = data_batch

        if cuda:
            samples, bindings, joins, targets = samples.cuda(), bindings.cuda(), joins.cuda(), targets.cuda()
            sample_masks, binding_masks, join_masks = sample_masks.cuda(), binding_masks.cuda(), join_masks.cuda()
        samples, bindings, joins, targets = Variable(samples), Variable(bindings), Variable(joins), Variable(
            targets)
        sample_masks, binding_masks, join_masks = Variable(sample_masks), Variable(binding_masks), Variable(
            join_masks)

        t = time.time()
        outputs = model(samples, bindings, joins, sample_masks, binding_masks, join_masks)
        t_total += time.time() - t

        for i in range(outputs.data.shape[0]):
            preds.append(outputs.data[i])

    return preds, t_total


def print_qerror(preds_unnorm, labels_unnorm):
    qerror = []
    for i in range(len(preds_unnorm)):
        if preds_unnorm[i] > float(labels_unnorm[i]):
            qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
        else:
            qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

    print("Median: {}".format(np.median(qerror)))
    print("90th percentile: {}".format(np.percentile(qerror, 90)))
    print("95th percentile: {}".format(np.percentile(qerror, 95)))
    print("99th percentile: {}".format(np.percentile(qerror, 99)))
    print("Max: {}".format(np.max(qerror)))
    print("Mean: {}".format(np.mean(qerror)))


def train_and_predict(num_queries, num_epochs, batch_size, hid_units, cuda):
    # 加载训练集和验证集
    num_materialized_samples = 1000
    dicts, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_bindings, train_data, test_data = \
        get_train_datasets(num_queries, num_materialized_samples)
    predicate2vec, binding2vec, join2vec = dicts

    # Train model
    sample_feats = len(predicate2vec) + num_materialized_samples
    binding_feats = len(binding2vec) + 100
    join_feats = len(join2vec)

    model = SetConv(sample_feats, binding_feats, join_feats, hid_units)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if cuda:
        model.cuda()

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    model.train()
    for epoch in range(num_epochs):
        loss_total = 0.

        for batch_idx, data_batch in enumerate(train_data_loader):

            samples, bindings, joins, targets, sample_masks, binding_masks, join_masks = data_batch

            if cuda:
                samples, bindings, joins, targets = samples.cuda(), bindings.cuda(), joins.cuda(), targets.cuda()
                sample_masks, binding_masks, join_masks = sample_masks.cuda(), binding_masks.cuda(), join_masks.cuda()
            samples, bindings, joins, targets = Variable(samples), Variable(bindings), Variable(joins), Variable(
                targets)
            sample_masks, binding_masks, join_masks = Variable(sample_masks), Variable(binding_masks), Variable(
                join_masks)

            optimizer.zero_grad()
            outputs = model(samples, bindings, joins, sample_masks, binding_masks, join_masks)
            loss = qerror_loss(outputs, targets.float(), min_val, max_val)
            loss_total += loss.item()
            loss.backward()
            optimizer.step()

        logger.info('Epoch {}, loss: {}'.format(epoch, loss_total / len(train_data_loader)))

    # Get final training and validation set predictions
    preds_train, t_total = predict(model, train_data_loader, cuda)
    logger.info('Prediction time per training sample: {}'.format(t_total / len(labels_train) * 1000))

    preds_test, t_total = predict(model, test_data_loader, cuda)
    logger.info('Prediction time per validation sample: {}'.format(t_total / len(labels_test) * 1000))

    # Unnormalize
    preds_train_unnorm = unnormalize_labels(preds_train, min_val, max_val)
    labels_train_unnorm = unnormalize_labels(labels_train, min_val, max_val)

    preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val)
    labels_test_unnorm = unnormalize_labels(labels_test, min_val, max_val)

    # Print metrics
    logger.info("\nQ-Error training set:")
    print_qerror(preds_train_unnorm, labels_train_unnorm)

    logger.info("\nQ-Error validation set:")
    print_qerror(preds_test_unnorm, labels_test_unnorm)

    # Write predictions
    file_name = "results/predictions.csv"
    with open(file_name, "w") as f:
        for i in range(len(preds_test_unnorm)):
            f.write(str(preds_test_unnorm[i]) + "," + str(labels_test_unnorm[i]) + "\n")

    predict_joins('join0', 'workload/join0/join0', num_materialized_samples, predicate2vec, binding2vec, join2vec,
                  batch_size, cuda, min_val, max_val, model)

    predict_joins('join1', 'workload/join1/join1', num_materialized_samples, predicate2vec, binding2vec, join2vec,
                  batch_size, cuda, min_val, max_val, model)

    predict_joins('join2', 'workload/join2/join2', num_materialized_samples, predicate2vec, binding2vec, join2vec,
                  batch_size, cuda, min_val, max_val, model)

    predict_joins('extend', 'workload/extend/extend', num_materialized_samples, predicate2vec, binding2vec, join2vec,
                  batch_size, cuda, min_val, max_val, model)


def predict_joins(workload_name, file_name, num_materialized_samples, predicate2vec, binding2vec, join2vec, batch_size,
                  cuda, min_val, max_val, model):
    # Load data
    predicates, joins, bindings, samples, label = load_data(file_name, num_materialized_samples)

    # Get feature encoding and proper normalization
    samples_enc = encode_samples(predicates, samples, predicate2vec)
    bindings_enc, joins_enc = encode_data(bindings, joins, binding2vec, join2vec)
    label_norm, _, _ = normalize_labels(label)

    print("Number of test samples: {}".format(len(label_norm)))

    max_num_joins = max([len(j) for j in joins_enc])
    max_num_bindings = max([len(p) for p in bindings_enc])

    # Get test set predictions
    test_data = make_dataset(samples_enc, bindings_enc, joins_enc, label_norm, max_num_joins, max_num_bindings)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    preds_test, t_total = predict(model, test_data_loader, cuda)
    print("Prediction time per test sample: {}".format(t_total / len(label_norm) * 1000))

    # Unnormalize
    preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val)

    # Print metrics
    print("\nQ-Error " + workload_name + ":")
    print_qerror(preds_test_unnorm, label)

    # Write predictions
    file_name = "results/" + workload_name + ".csv"
    with open(file_name, "w") as f:
        for i in range(len(preds_test_unnorm)):
            f.write(str(preds_test_unnorm[i]) + "," + str(label[i]) + "\n")


if __name__ == "__main__":
    train_and_predict(
        num_queries=100000,
        num_epochs=100,
        batch_size=1024,
        hid_units=256,
        cuda=False
    )
