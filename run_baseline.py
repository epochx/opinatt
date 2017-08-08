#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import time
import sys
import subprocess
import os
import random
import json
import hashlib
import copy

from enlp.rnn.theano_elman_rnn import ElmanRNN
from enlp.rnn.theano_jordan_rnn import JordanRNN
from enlp.rnn.theano_lstm import LSTM
from enlp.rnn.theano_elman_brnn import ElmanBRNN
from enlp.rnn.theano_blstm import BLSTM

from enlp.rnn.utils import contextwin, minibatch, shuffle
from enlp.eval import conlleval, classeval, tagclasses2classes, classes2class


def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  if v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


def get_model_id(args):
  params = vars(args)
  sha = hashlib.sha1(str(params)).hexdigest()
  return sha

def run_evaluation(token_indices, predicted_indices, gold_indices, idx2token, idx2label,
                   simpleidx2label=None, out_file=None):

    tokens = [map(lambda x: idx2token[x], w) for w in token_indices]


    predictions = [map(lambda x: idx2label[x], idx_prediction)
                   for idx_prediction  in predicted_indices]
    ground_truth = [map(lambda x: idx2label[x], y) for y in gold_indices]

    result = conlleval(predictions, ground_truth, tokens, out_file)
    tag_result = None
    class_result = None

    if simpleidx2label:
        tag_predictions = []
        class_predictions = []
        tag_golds = []
        class_golds = []

        for gold_sequence in gold_indices:
            seq_tags, seq_classes = zip(*map(lambda x: simpleidx2label[str(x)], gold_sequence))
            tag_golds.append(seq_tags)
            class_golds += tagclasses2classes(seq_classes, seq_tags)

        for i, predicted_sequence in enumerate(predicted_indices):
            seq_tags, seq_classes = zip(*map(lambda x: simpleidx2label[str(x)], predicted_sequence))
            tag_predictions.append(seq_tags)
            class_predictions += tagclasses2classes(seq_classes, tag_golds[i])

        tag_result = conlleval(tag_predictions, tag_golds, tokens, out_file + ".simple")

        class_out_file = out_file.replace("tagging", "classification") +  ".simple"

        labels = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
        class_result = classeval(class_predictions, class_golds, labels, class_out_file)

    return result, tag_result, class_result



rnns = {"ElmanRNN": ElmanRNN,
        "JordanRNN": JordanRNN,
        "LSTM": LSTM,
        "ElmanBRNN": ElmanBRNN,
        "BLSTM": BLSTM}

# parse parameters
desc = "Help for train.py, a script that trains RNNs for opinion mining given a JSON file"

parser = argparse.ArgumentParser(description=desc)

parser.add_argument('--json_path',
                    required=True,
                    help="Path to the JSON file")

parser.add_argument('--results_path',
                    required=True,
                    help="Path to store results")

parser.add_argument('--rnn_type',
                    choices=rnns,
                    default="LSTM",
                    help="Type of RNN to use")

parser.add_argument('-ne', "--num_epochs",
                    type=int,
                    default=5,
                    help="Number of epochs")

parser.add_argument('-ws', '--window_size',
                    type=int,
                    default=3,
                    help="Word window context size")

parser.add_argument('-nh','--num_hidden',
                    type=int,
                    default=200,
                    help="Size of hidden state")

parser.add_argument('-bs', '--batch_size',
                    type=int,
                    default=6,
                    help="Batch size/ truncated-BPTT steps")

parser.add_argument('-lr', '--learning_rate',
                    default=0.01,
                    type=float,
                    help="Learning rate")

parser.add_argument("-s", '--seed',
                    type=int,
                    default=123,
                    help="Random seed")

parser.add_argument("--fine_tuning", "-ft",
                    type=str2bool,
                    default=True,
                    help="Add fine tuning")

parser.add_argument("--feat", "-f",
                    type=str2bool,
                    default=True,
                    help="Use linguistic features")

parser.add_argument("--decay", "-d",
                    type=str2bool,
                    default=True,
                    help="Use decay when no improvement")

parser.add_argument("--sentence_train", "-st",
                    type=str2bool,
                    default=False,
                    help="Train using sentence NLL (otherwise use batches)")

parser.add_argument("--verbose", "-v",
                    type=str2bool,
                    default=True,
                    help="Show training progress and results")

parser.add_argument("--weight", "-w",
                    default=1.0,
                    type=float,
                    help="Weight for domain adaptation")

parser.add_argument("--overwrite",
                    default=False,
                    type=str2bool,
                    help="Rewrite trained model")

args = parser.parse_args()

rnn_type = args.rnn_type
ne = args.num_epochs
ws = args.window_size
nh = args.num_hidden
bs = args.batch_size
lr = args.learning_rate
seed = args.seed
ft = args.fine_tuning
feat = args.feat
decay = args.decay
st = args.sentence_train
v = args.verbose
w = args.weight

folds = False

for k, v in vars(args).iteritems():
    print('%s: %s' % (k, str(v)))

if os.path.isdir(args.json_path):
    if args.json_path.endswith("/"):
        args.json_path = args.json_path[:-1]

    json_base_path, json_path = os.path.split(args.json_path)

    paths = [(json_base_path, json_path, json_filename)
             for json_filename in sorted(os.listdir(args.json_path))]
    folds = True

elif os.path.isfile(args.json_path):
    json_base_dir, json_filename = os.path.split(args.json_path)
    paths = [(json_base_dir, "", json_filename) ]

# containers for averaging k folds
best_f1s_test, best_f1s_valid = [], []

base_results_path = args.results_path

for counter, (json_base_path, json_path, json_filename) in enumerate(paths):

    current_args = copy.copy(args)

    model_id = get_model_id(current_args)

    current_args.json_path = os.path.join(*(json_base_path,
                                            json_path,
                                            json_filename))

    current_args.results_path = os.path.join(*(base_results_path,
                                               json_path,
                                               json_filename))

    results_dir = os.path.join(current_args.results_path, model_id)

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    else:
        if args.overwrite:
            print("Model already trained, overwriting")
        else:
            print("Model already trained. Set overwrite=True to overwrite\n")
            continue

    # store parameters
    with open(os.path.join(results_dir, "params.json"), "w") as f:
        params = vars(current_args)
        params["model_id"] = model_id
        json.dump(params, f)

    has_test, has_valid, has_domain = False, False, False

    # load the dataset
    corpus = json.load(open(current_args.json_path, "rb"))

    token2idx = corpus["token2idx"]
    label2idx = corpus["tag2idx"]
    simpleidx2label = None

    if len(label2idx) >  4:
        simpleidx2label = corpus["simple_rev_tag_vocab"]

    # check if corpus has validation/test data
    # which are not included when training for PRED domain adaptation
    if 'valid_x' in corpus:
        has_valid = True

    if 'test_x' in corpus:
        has_test = True

    if 'domain' in corpus:
        has_domain = True

    idx2label = dict((k, v) for v, k in label2idx.iteritems())
    idx2token = dict((k, v) for v, k in token2idx.iteritems())

    train_x, train_y = corpus['train_x'], corpus['train_y']
    train_x_feat = [np.asarray(m, dtype=np.float32)
                    for m in corpus['train_feat_x']]

    if has_domain and w < 1.0:
        train_weights = np.asarray(corpus['train_weights'], dtype=np.float32)
        train_weights = train_weights * w + (1 - train_weights)
    else:
        train_weights = np.ones((len(train_x,)), dtype=np.float32)

    if has_valid:
        valid_x, valid_y = corpus['valid_x'], corpus['valid_y']
        valid_x_feat = [np.asarray(m, dtype=np.float32)
                        for m in corpus['valid_feat_x']]
        if has_domain and w < 1.0:
            valid_weights = np.asarray(corpus['valid_weights'], dtype=np.float32)
            valid_weights = valid_weights * w + (1 - valid_weights)
        else:
            valid_weights = np.ones((len(valid_x, )), dtype=np.float32)

    if has_test:
        test_x, test_y = corpus['test_x'], corpus['test_y']
        test_x_feat = [np.asarray(m, dtype=np.float32)
                       for m in corpus['test_feat_x']]
        if has_domain and w < 1.0:
            test_weights = np.asarray(corpus['test_weights'], dtype=np.float32)
            test_weights = test_weights * w + (1 - test_weights)
        else:
            test_weights = np.ones((len(test_x, )), dtype=np.float32)

    embeddings = np.asarray(corpus['embeddings_matrix'], dtype=np.float32)

    if has_valid and has_test:
        # vocabulary size
        # vs = len(set(reduce(lambda x, y: list(x) + list(y), train_x + valid_x + test_x)))

        # number of classes

        nc = len(set(reduce(lambda x, y: list(x) + list(y), train_y + valid_y + test_y))) + 1

    else:
        vs = len(set(reduce(lambda x, y: list(x) + list(y), train_x)))
        nc = len(set(reduce(lambda x, y: list(x) + list(y), train_y))) + 1


    # number of sentences
    ns = len(train_x)

    # feat dim
    if feat:
        fd = corpus["num_binary_features"]
    else:
        fd = 0

    # instantiate the model
    np.random.seed(seed)
    random.seed(seed)

    # init model

    RNN = rnns.get(rnn_type, None)
    if RNN is None:
        raise Exception("Wrong RNN type provided.")

    rnn = RNN(nh, nc, ws, embeddings, featdim=fd,
              fine_tuning=ft, truncate_gradient=bs)

    # TRAINING (with early stopping on validation set)
    best_test = {'f1': -np.inf }
    best_valid = {'f1': -np.inf }
    tag_res_valid = None
    tag_res_test = None

    tag_simple_res_valid = None
    class_simple_res_valid = None
    tag_simple_res_test = None
    class_simple_res_test = None


    best_epoch = 0

    tagging_valid_out_file = os.path.join(results_dir, 'tagging.valid.hyp.txt')
    tagging_test_out_file = os.path.join(results_dir, 'tagging.test.hyp.txt')


    log = []

    for e in xrange(ne):
        shuffle([train_x, train_x_feat, train_y, train_weights], seed)
        tic = time.time()
        nlls = []
        for i in xrange(len(train_x)):
        #for i in xrange(20):
            cwords = contextwin(train_x[i], ws)
            labels = train_y[i]
            weight = train_weights[i]
            if st:
                features = train_x_feat[i]
                words = map(lambda x: np.asarray(x).astype('int32'), cwords)
                sentence_nll = rnn.sentence_train(words, features, labels, lr, weight)
                nlls.append(sentence_nll)
                if ft:
                    rnn.normalize()

            else:
                words = map(lambda x: np.asarray(x).astype('int32'),
                            minibatch(cwords, bs))
                features = minibatch(train_x_feat[i], bs)
                bnlls = []
                for word_batch, feat_batch, label_last_word in zip(words, features, labels):
                    nll = rnn.train(word_batch, feat_batch, label_last_word, lr, weight)
                    bnlls.append(nll)
                    if ft:
                        rnn.normalize()
                nlls.append(np.mean(bnlls))

            if v:
                print '[learning] epoch %i >> %2.2f%%' % (e, (i + 1) * 100. / ns),\
                       'completed in %.2f (sec) <<\r' % (time.time() - tic),
                sys.stdout.flush()

        print ""

        if has_valid:
            idx_predictions_valid = [rnn.classify(np.asarray(contextwin(x, ws)).astype('int32'), feat_x, valid_weight)
                                     for x, feat_x, valid_weight in zip(valid_x, valid_x_feat, valid_weights)]

            tag_res_valid, tag_simple_res_valid, class_simple_res_valid  = run_evaluation(
                valid_x, idx_predictions_valid, valid_y, idx2token, idx2label,
                simpleidx2label=simpleidx2label, out_file=tagging_valid_out_file)
            print("  %s tagging f1-score: %.2f" % ("Validation", tag_res_valid['f1']))

            if simpleidx2label:
                print("  %s simple tagging f1-score: %.2f" % ("Validation", tag_simple_res_valid['f1']))
                for label, cm in class_simple_res_valid.items():
                    print("  \tsimple %s accuracy: %.2f  precision: %.2f  recall: %.2f  f1-score: %.2f" % (
                        label, 100 * cm["a"], 100 * cm["p"], 100 * cm["r"], 100 * cm["f1"]))

        if has_test:
            idx_predictions_test = [rnn.classify(np.asarray(contextwin(x, ws)).astype('int32'), feat_x, test_weight)
                                    for x, feat_x, test_weight in zip(test_x, test_x_feat, test_weights)]

            tag_res_test, tag_simple_res_test, class_simple_res_test  = run_evaluation(
                test_x, idx_predictions_test, test_y, idx2token, idx2label,
                simpleidx2label=simpleidx2label, out_file=tagging_test_out_file)
            print("  %s tagging f1-score: %.2f" % ("Test", tag_res_test['f1']))

            if simpleidx2label:
                print("  %s simple tagging f1-score: %.2f" % ("Test", tag_simple_res_test['f1']))
                for label, cm in class_simple_res_test.items():
                    print("  \tsimple %s accuracy: %.2f  precision: %.2f  recall: %.2f  f1-score: %.2f" % (
                        label, 100 * cm["a"], 100 * cm["p"], 100 * cm["r"], 100 * cm["f1"]))

        # do stuff when provided valid and test data
        if has_valid and has_test:

            if tag_res_valid['f1'] > best_valid['f1']:
                rnn.save(results_dir)
                best_valid = tag_res_valid
                best_test = tag_res_test
                best_epoch = e
                if v:
                    print 'NEW BEST: epoch', e, 'valid F1', best_valid['f1'], 'best test F1', best_test['f1'], ' '*20

                subprocess.call(['mv', tagging_valid_out_file,  tagging_valid_out_file + '.best'])
                subprocess.call(['mv', tagging_test_out_file,  tagging_test_out_file + '.best'])
            else:
                print ''

        # learning rate decay if no improvement in 5 epochs
            if decay and abs(best_epoch-e) >= 5:
                lr *= 0.5

            if lr < 1e-5:
                break

        # if no valid/test data provided, just save the model every epoch
        else:
            rnn.save(results_dir)

        log.append([tag_res_valid, tag_simple_res_valid, class_simple_res_valid,
                    tag_res_test, tag_simple_res_test, class_simple_res_test])

        with open(os.path.join(results_dir, "log.json"), "w") as f:
            json.dump(log, f)