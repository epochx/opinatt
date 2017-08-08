#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os
import subprocess
import sys
import time
import hashlib
import argparse
import copy
import re

import numpy as np
import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin

from enlp.eval import conlleval, classeval, tagclasses2classes, classes2class
from enlp.multi_task_rnn import MultiTaskModel



def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  if v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


def main():

  parser = argparse.ArgumentParser()


  parser.add_argument("--learning_rate",
                      type=float,
                      default=0.5,
                      help="(initial) Learning rate.")


  parser.add_argument("--decay_factor",
                      type=float,
                      default=0.9,
                      help="Learning rate decay factor")

  parser.add_argument("--early_stopping",
                      type=int,
                      default=1000,
                      help="Stop training if no improvement")

  parser.add_argument("--optimizer",
                      type=str,
                      default="adam",
                      help="Optimizer")

  parser.add_argument("--max_gradient_norm",
                      type=float,
                      default=5.0,
                      help="Clip gradients to this norm.")

  parser.add_argument("--batch_size",
                      type=int,
                      default=16,
                      help="Batch size to use during training.")

  parser.add_argument("--eval_batch_size",
                      type=int,
                      default=16,
                      help="Batch size to use during evaluation.")

  parser.add_argument("--size",
                      default=100,
                      type=int,
                      help="Size of each model layer.")

  parser.add_argument("--word_embedding_size",
                      default=0,
                      type=int,
                      help="Size of the word embedding. Use 0 to use json data provided.")

  parser.add_argument("--num_layers",
                      default=1,
                      type=int,
                      help="Number of layers in the model.")

  parser.add_argument("--json_path",
                      required=True,
                      type=str,
                      help="JSON Data dir (for folds) or file path. Checks JSON_DIR first.")

  parser.add_argument("--results_path",
                      default="",
                      help="Path to save trained model and outputs. Checks RESULTS_DIR first")

  parser.add_argument("--checkpoint_path",
                      default="",
                      type=str,
                      help="Path to load trained model")

  parser.add_argument("--steps_per_checkpoint",
                      default=100,
                      type=int,
                      help="How many training steps to do per checkpoint.")


  parser.add_argument("--max_training_steps",
                      default=10000,
                      type=int,
                      help="Max training steps.")

  parser.add_argument("--use_attention",
                      default=True,
                      type=str2bool,
                      help="Use attention based RNN")

  parser.add_argument("--max_sequence_length",
                      default=200,
                      type=int,
                      help="Max sequence length.")

  parser.add_argument("--context_win_size",
                      default=3,
                      type=int,
                      help="Context window size.")

  parser.add_argument("--dropout_keep_prob",
                      default=0.5 ,
                      type=float,
                      help="dropout keep cell input and output prob.")

  parser.add_argument("--zoneout_keep_prob",
                      default=1 ,
                      type=float,
                      help="zoneout keep cell input and output prob.")

  parser.add_argument("--bidirectional_rnn",
                      default=True,
                      type=str2bool,
                      help="Use bidirectional RNN")

  parser.add_argument("--use_previous_output",
                      default=False,
                      type=str2bool,
                      help="Use previous predicted label y_{t-1}")

  parser.add_argument("--use_binary_features",
                      default=True,
                      type=str2bool,
                      help="Use binary features")

  parser.add_argument("--overwrite",
                      default=False,
                      type=str2bool,
                      help="Rewrite trained model")

  parser.add_argument("--rnn",
                      default="lstm",
                      choices=["lstm", "gru", "bnlstm"],
                      help="Rnn cell type",
                      type=str)

  FLAGS = parser.parse_args()

  _buckets = [(FLAGS.max_sequence_length, FLAGS.max_sequence_length)]

  if FLAGS.max_sequence_length == 0:
    print('Please indicate max sequence length. Exit')
    exit()

  if FLAGS.zoneout_keep_prob < 1 and FLAGS.dropout_keep_prob < 1:
    print('Please choose dropout for zoneout. Exit')
    exit()

  task = {'intent': 0, 'tagging': 1, 'joint': 0}

  if "joint" in FLAGS.json_path:
    task['intent'] = 1
    task['joint'] = 1

  FLAGS.task = task

  json_dir = os.environ.get('JSON_DIR', None)
  results_dir = os.environ.get('RESULTS_DIR', None)

  if json_dir:
    FLAGS.json_path = os.path.join(json_dir, FLAGS.json_path)

  if results_dir:
    FLAGS.results_path = os.path.join(results_dir, FLAGS.results_path)

  if os.path.isdir(FLAGS.json_path):
    if FLAGS.json_path.endswith("/"):
      FLAGS.json_path = FLAGS.json_path[:-1]
    json_base_path, json_path = os.path.split(FLAGS.json_path)

    paths = [(json_base_path, json_path, json_filename)
             for json_filename in sorted(os.listdir(FLAGS.json_path))]

  elif os.path.isfile(FLAGS.json_path):
    json_base_dir, json_filename = os.path.split(FLAGS.json_path)
    paths = [(json_base_dir, "", json_filename) ]
  else:
    print("Not valid JSON path")
    exit()

  BASE_FLAGS=FLAGS

  base_results_path = FLAGS.results_path

  for json_base_path, json_path, json_filename in paths:

    FLAGS = copy.copy(FLAGS)
    # we obtain model_id before modifying paths, so the same model keeps the id for different folds
    model_id = get_model_id(FLAGS)

    # if folds this is json_path/
    FLAGS.json_path = os.path.join(*(json_base_path,
                                     json_path,
                                     json_filename))

    FLAGS.results_path = os.path.join(*(base_results_path,
                                        json_path,
                                        json_filename))

    if not os.path.isdir(FLAGS.results_path):
      os.makedirs(FLAGS.results_path)

    print('Applying Parameters:')
    for k, v in vars(FLAGS).iteritems():
      print('%s: %s' % (k, str(v)))


    results_dir = os.path.join(FLAGS.results_path, model_id)

    if not os.path.isdir(results_dir):
      os.makedirs(results_dir)
    else:
      if FLAGS.overwrite:
        print("Model already trained, overwriting")
      else:
        print("Model already trained. Set overwrite=True to overwrite\n")
        continue

    # store parameters
    with open(os.path.join(results_dir, "params.json"), "w") as f:
      json.dump(vars(FLAGS), f)

    log = []

    tagging_valid_out_file = os.path.join(results_dir, 'tagging.valid.hyp.txt')
    tagging_test_out_file = os.path.join(results_dir, 'tagging.test.hyp.txt')

    class_valid_out_file = os.path.join(results_dir, 'classification.valid.hyp.txt')
    class_test_out_file = os.path.join(results_dir, 'classification.test.hyp.txt')

    att_valid_out_file = os.path.join(results_dir, 'attentions.valid.hyp.json')
    att_test_out_file = os.path.join(results_dir, 'attentions.test.hyp.json')

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
      # Read data into buckets and compute their sizes.
      print("Reading train/valid/test data")
      train, valid, test, vocabs, rev_vocabs, binary_feat_dim, embeddings_matrix = read_data(FLAGS)

      train_x, train_feat_x, train_y, train_z = train
      valid_x, valid_feat_x, valid_y, valid_z = valid
      test_x, test_feat_x, test_y, test_z = test
      vocab, tag_vocab, label_vocab = vocabs
      rev_vocab, rev_tag_vocab, rev_label_vocab, extra_rev_vocab = rev_vocabs

      print(len(train_x))
      print(len(valid_x))
      print(len(test_x))

      train_set = generate_buckets(_buckets, train_x, train_y, train_z, extra_x=train_feat_x)
      dev_set = generate_buckets(_buckets, valid_x, valid_y, valid_z, extra_x=valid_feat_x)
      test_set = generate_buckets(_buckets, test_x, test_y, test_z, extra_x=test_feat_x)

      train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
      train_total_size = float(sum(train_bucket_sizes))

      train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                             for i in xrange(len(train_bucket_sizes))]

      # Create model.
      print("Max sequence length: %d." % _buckets[0][0])
      print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))

      model, model_test = create_model(sess, len(vocab), len(tag_vocab),
                                       len(label_vocab), embeddings_matrix, binary_feat_dim,
                                       FLAGS, _buckets)

      # train_writer = tf.summary.FileWriter(results_dir, sess.graph)
      print("Creating model with source_vocab_size=%d, target_vocab_size=%d, and label_vocab_size=%d." %
            (len(vocab), len(tag_vocab), len(label_vocab)))

      # This is the training loop.
      step_time, loss = 0.0, 0.0
      current_step = 0
      best_step = 0

      best_valid_tagging_score = 0
      best_test_tagging_score = 0
      best_valid_class_score = 0
      best_test_class_score = 0

      while model.global_step.eval() < FLAGS.max_training_steps:
        random_number_01 = np.random.random_sample()
        bucket_id = min([i for i in xrange(len(train_buckets_scale))
                         if train_buckets_scale[i] > random_number_01])

        # Get a batch and make a step.
        start_time = time.time()
        encoder_inputs, _, encoder_extra_inputs, tags, tag_weights, batch_sequence_length, labels = \
          model.get_batch(train_set, bucket_id)

        if task['joint'] == 1:
          output = model.joint_step(sess, encoder_inputs, encoder_extra_inputs, tags, tag_weights, labels,
                                    batch_sequence_length, bucket_id, False)
          _, step_loss, tagging_logits, tagging_att, class_logits, class_att = output

        elif task['tagging'] == 1:
          output = model.tagging_step(sess, encoder_inputs, encoder_extra_inputs, tags, tag_weights,
                                      batch_sequence_length, bucket_id, False)
          _, step_loss, tagging_logits, tagging_att = output
        elif task['intent'] == 1:
          output = model.classification_step(sess, encoder_inputs, encoder_extra_inputs,
                                             labels, batch_sequence_length, bucket_id, False)
          _, step_loss, class_logits, class_att = output

        step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
        loss += step_loss / FLAGS.steps_per_checkpoint
        current_step += 1

        # Once in a while, we save checkpoint, print statistics, and run evals.
        if current_step % FLAGS.steps_per_checkpoint == 0:
          perplexity = math.exp(loss) if loss < 300 else float('inf')
          print("global step %d step-time %.2f. Training perplexity %.2f"
                % (model.global_step.eval(), step_time, perplexity))
          sys.stdout.flush()
          # Save checkpoint and zero timer and loss.
          checkpoint_path = os.path.join(results_dir, "model.ckpt")
          step_time, loss = 0.0, 0.0

          # valid
          valid_class_result, valid_tagging_result, valid_extra_result = \
            run_valid_test(model_test, sess, dev_set, 'Valid', task,
                           rev_vocab, rev_tag_vocab, rev_label_vocab, FLAGS, _buckets,
                           tagging_out_file=tagging_valid_out_file,
                           classification_out_file=class_valid_out_file,
                           extra_rev_vocab=extra_rev_vocab,
                           attentions_out_file=att_valid_out_file)

          if valid_tagging_result['f1'] > best_valid_tagging_score:
            best_valid_tagging_score = valid_tagging_result['f1']
            # save the best output file
            subprocess.call(['mv', tagging_valid_out_file, tagging_valid_out_file + '.best'])

            if task['joint'] == 1 and valid_class_result["all"]["a"] > best_valid_class_score:
              best_valid_class_score = valid_class_result["all"]["a"]
              subprocess.call(['mv', class_valid_out_file, class_valid_out_file + '.best'])
              subprocess.call(['mv', att_valid_out_file, att_valid_out_file + '.best'])

            model.saver.save(sess, checkpoint_path)
            best_step = current_step

          # test
          test_class_result, test_tagging_result, test_extra_result = \
            run_valid_test(model_test, sess, test_set, 'Test', task,
                           rev_vocab, rev_tag_vocab, rev_label_vocab, FLAGS, _buckets,
                           tagging_out_file=tagging_test_out_file,
                           classification_out_file=class_test_out_file,
                           extra_rev_vocab=extra_rev_vocab,
                           attentions_out_file=att_test_out_file)

          if task['tagging'] == 1 and test_tagging_result['f1'] > best_test_tagging_score:
            best_test_tagging_score = test_tagging_result['f1']
            # save the best output file
            subprocess.call(['mv', tagging_test_out_file, tagging_test_out_file + '.best'])

            if task['joint'] == 1 and test_class_result["all"]["a"] > best_test_class_score:
              best_test_class_score = test_class_result["all"]["a"]
              subprocess.call(['mv', class_test_out_file, class_test_out_file + '.best'])
            subprocess.call(['mv', att_test_out_file, att_test_out_file + '.best'])

            model.saver.save(sess, checkpoint_path)

          log.append([valid_class_result, valid_tagging_result, valid_extra_result,
                      test_class_result, test_tagging_result, test_extra_result])

          with open(os.path.join(results_dir, "log.json"), "w") as f:
            json.dump(log, f)

          if FLAGS.early_stopping and current_step - best_step > FLAGS.early_stopping:
            break

    tf.reset_default_graph()

    FLAGS = BASE_FLAGS

def generate_buckets(_buckets, x, y, z, extra_x=None):
  data_set = [[] for _ in _buckets]
  for i, (source, target, label) in enumerate(zip(x, y, z)):
    for bucket_id, (source_size, target_size) in enumerate(_buckets):
      if len(source) < source_size and len(target) < target_size:
        if extra_x:
            data_set[bucket_id].append([source, extra_x[i], target, label])
        else:
            data_set[bucket_id].append([source, None, target, label])
        break
  return data_set # 4 outputs in each unit: source_ids, source_matrix, target_ids, label_ids



def get_model_id(FLAGS):
  params = vars(FLAGS)
  sha = hashlib.sha1(str(params)).hexdigest()
  return sha


def read_data(FLAGS):
  """
  Reads the json and extracts the corresponding data

  :param json_path:
  :param task:
  :param FLAGS:
  :return:
  """
  with open(FLAGS.json_path, "r") as f:

    jsondic = json.load(f)

    len_train_x = len(jsondic["train_x"])
    len_valid_x = len(jsondic["valid_x"])
    len_test_x = len(jsondic["test_x"])

    print("Train examples: " + str(len_train_x))
    print("Valid examples: " + str(len_valid_x))
    print("Test examples: " + str(len_test_x))

    if len_test_x % FLAGS.eval_batch_size:
      #print("Reseting eval_batch_size to 1")
      #FLAGS.eval_batch_size = 1
      test_limit = len_test_x - (len_test_x % FLAGS.eval_batch_size)
      print("Incompatible eval_batch_size for testing, reseting to " + str(test_limit))
    else:
      test_limit = len_test_x

    if len_valid_x % FLAGS.eval_batch_size:
      valid_limit = len_valid_x - (len_valid_x % FLAGS.eval_batch_size)
      print("Incompatible eval_batch_size for validation, reseting to " + str(valid_limit))
    else:
      valid_limit = len_valid_x

    train_x, train_y = jsondic["train_x"], jsondic["train_y"]
    valid_x, valid_y = jsondic["valid_x"][:valid_limit], jsondic["valid_y"][:valid_limit]
    test_x, test_y = jsondic["test_x"][:test_limit], jsondic["test_y"][:test_limit]

    if FLAGS.task['intent'] == 1:
      train_z = jsondic["train_z"]
      valid_z = jsondic["valid_z"][:valid_limit]
      test_z = jsondic["test_z"][:test_limit]

      label_vocab = jsondic["class2idx"]
      rev_label_vocab = {idx: token for token, idx in label_vocab.items()}

    else:
      # generating "fake" labels
      train_z = [0 for seq in train_x]
      valid_z = [0 for seq in valid_x]
      test_z = [0 for seq in test_x]
      label_vocab = rev_label_vocab = {}

    if FLAGS.word_embedding_size == 0:
      if "embeddings_matrix" in jsondic:
        print("Loading embeddings from JSON data...")
        embeddings_matrix = np.asarray(jsondic["embeddings_matrix"], dtype=np.float32)
      else:
        print('No embeddings in JSON data, please indicate word embedding size')
        exit()
    else:
      embeddings_matrix = None

    if FLAGS.use_binary_features:
      if jsondic["num_binary_features"] > 0:
        print("Found binary features, adding them.")
        train_feat_x = jsondic["train_feat_x"]
        valid_feat_x = jsondic["valid_feat_x"][:valid_limit]
        test_feat_x = jsondic["test_feat_x"][:test_limit]
        binary_feat_dim = jsondic["num_binary_features"]
      else:
        print("No binary feats found in JSON data, please change flag an re run.")
        exit()
    else:
      train_feat_x = valid_feat_x = test_feat_x = None
      binary_feat_dim = 0

    vocab = jsondic["token2idx"]
    rev_vocab = {idx: token for token, idx in vocab.items()}

    tag_vocab = jsondic["tag2idx"]
    rev_tag_vocab = {idx: token for token, idx in tag_vocab.items()}

    train = [train_x, train_feat_x, train_y, train_z]
    valid = [valid_x, valid_feat_x, valid_y, valid_z]
    test = [test_x, test_feat_x, test_y, test_z]
    vocabs = [vocab, tag_vocab, label_vocab]

    if FLAGS.task["joint"] == 1:
        extra_rev_vocab = jsondic["joint_rev_tag_vocab"]
    elif FLAGS.task["tagging"] == 1 and len(tag_vocab) > 4:
        extra_rev_vocab = jsondic["simple_rev_tag_vocab"]
    else:
        extra_rev_vocab = None

    rev_vocabs = [rev_vocab, rev_tag_vocab, rev_label_vocab, extra_rev_vocab]

    return [train, valid, test, vocabs, rev_vocabs, binary_feat_dim, embeddings_matrix]


def create_model(session, source_vocab_size, target_vocab_size, label_vocab_size,
                 word_embedding_matrix, binary_feat_dim, FLAGS, _buckets):
  """Create model and initialize or load parameters in session."""
  with tf.variable_scope("model", reuse=None):
    model_train = MultiTaskModel(source_vocab_size, target_vocab_size, label_vocab_size, _buckets,
                                 FLAGS.word_embedding_size, FLAGS.size, FLAGS.num_layers,
                                 FLAGS.max_gradient_norm, FLAGS.batch_size,
                                 dropout_keep_prob=FLAGS.dropout_keep_prob,
                                 zoneout_keep_prob=FLAGS.zoneout_keep_prob,
                                 rnn=FLAGS.rnn,
                                 forward_only=False, use_attention=FLAGS.use_attention,
                                 bidirectional_rnn=FLAGS.bidirectional_rnn, task=FLAGS.task,
                                 context_win_size=FLAGS.context_win_size,
                                 word_embedding_matrix=word_embedding_matrix,
                                 learning_rate=FLAGS.learning_rate,
                                 decay_factor=FLAGS.decay_factor,
                                 use_previous_output=FLAGS.use_previous_output,
                                 use_binary_features=FLAGS.use_binary_features,
                                 binary_feat_dim=binary_feat_dim,
                                 optimizer=FLAGS.optimizer)

  with tf.variable_scope("model", reuse=True):
    model_test = MultiTaskModel(source_vocab_size, target_vocab_size, label_vocab_size, _buckets,
                                FLAGS.word_embedding_size, FLAGS.size, FLAGS.num_layers,
                                FLAGS.max_gradient_norm, FLAGS.eval_batch_size,
                                dropout_keep_prob=FLAGS.dropout_keep_prob,
                                zoneout_keep_prob=FLAGS.zoneout_keep_prob,
                                rnn=FLAGS.rnn,
                                forward_only=True, use_attention=FLAGS.use_attention,
                                bidirectional_rnn=FLAGS.bidirectional_rnn, task=FLAGS.task,
                                context_win_size=FLAGS.context_win_size,
                                word_embedding_matrix=word_embedding_matrix,
                                learning_rate=FLAGS.learning_rate,
                                decay_factor=FLAGS.decay_factor,
                                use_previous_output=True,
                                use_binary_features=FLAGS.use_binary_features,
                                binary_feat_dim=binary_feat_dim,
                                optimizer=FLAGS.optimizer)

  ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)

  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model_train.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
    if FLAGS.word_embedding_size == 0 and word_embedding_matrix is not None:
      with tf.variable_scope("model", reuse=True):
        embedding = tf.get_variable("embedding")
        session.run(embedding.assign(word_embedding_matrix))
  return model_train, model_test




def run_valid_test(model_test, sess, data_set, mode, task,
                   rev_vocab, rev_tag_vocab, rev_label_vocab, FLAGS, _buckets,
                   tagging_out_file=None, classification_out_file=None,
                   extra_rev_vocab=None, attentions_out_file=None):
    # modes: Eval, Test
    # Run evals on development/test set and print the accuracy.
    word_idx_list = []
    ref_tag_idx_list, hyp_tag_idx_list = [], []
    ref_label_idx_list, hyp_label_idx_list = [], []
    hyp_label_attentions = []
    hyp_class_attentions = []
    model_batch_size = model_test.batch_size
    for bucket_id in xrange(len(_buckets)):
        eval_loss = 0.0
        count = 0
        while count < len(data_set[bucket_id]):
            #if model_batch_size == 1:
            #  model_inputs = model_test.get_one(data_set, bucket_id, count)
            #else:
            model_inputs = model_test.get_batch(data_set, bucket_id, count)

            encoder_inputs, simple_encoder_inputs, encoder_extra_inputs, \
              tags, tag_weights, sequence_lengths, labels = model_inputs

            tagging_logits = []
            class_logits = []

            if task['joint'] == 1:
              output = model_test.joint_step(sess, encoder_inputs, encoder_extra_inputs, tags, tag_weights,
                                             labels, sequence_lengths, bucket_id, True)
              _, step_loss, tagging_logits, tagging_att, class_logits, class_att = output

            elif task['tagging'] == 1:
              output = model_test.tagging_step(sess, encoder_inputs, encoder_extra_inputs,
                                      tags, tag_weights, sequence_lengths, bucket_id, True)
              _, step_loss, tagging_logits, tagging_att = output

            #elif task['intent'] == 1:
            #  output = model_test.classification_step(sess, encoder_inputs, encoder_extra_inputs,
            #                                          labels, sequence_lengths, bucket_id, True)
            #  _, step_loss, class_logits, class_att = output



            eval_loss += step_loss / len(data_set[bucket_id])

            if task['intent'] == 1:
              for seq_id, sequence_length in enumerate(sequence_lengths):
                ref_label_idx = labels[0][seq_id]
                ref_label_idx_list.append(ref_label_idx)
                hyp_label_idx = np.argmax(class_logits[seq_id])
                hyp_label_idx_list.append(hyp_label_idx)
                hyp_class_attentions.append(class_att[seq_id][:sequence_length].tolist())

            if task['tagging'] == 1:

              for seq_id, sequence_length in enumerate(sequence_lengths):

                current_word_idx_list = []
                current_ref_tag_idx_list = []
                current_hyp_tag_idx_list = []
                current_att_list = []
                for token_id in xrange(sequence_length):
                  current_word_idx_list.append(simple_encoder_inputs[token_id][seq_id])
                  current_ref_tag_idx_list.append(tags[token_id][seq_id])
                  current_hyp_tag_idx_list.append(np.argmax(tagging_logits[token_id][seq_id]))
                  current_att_list.append(tagging_att[token_id][seq_id][:sequence_length].tolist())

                word_idx_list.append(current_word_idx_list)
                ref_tag_idx_list.append(current_ref_tag_idx_list)
                hyp_tag_idx_list.append(current_hyp_tag_idx_list)
                hyp_label_attentions.append(current_att_list)

            count += model_batch_size

    word_list = [[rev_vocab[idx] for idx in sequence] for sequence in word_idx_list]

    extra_eval_result = None
    classification_eval_result = None

    hyp_tag_list = [[rev_tag_vocab[idx] for idx in sequence] for sequence in hyp_tag_idx_list]
    ref_tag_list = [[rev_tag_vocab[idx] for idx in sequence] for sequence in ref_tag_idx_list]

    tagging_eval_result = conlleval(hyp_tag_list, ref_tag_list, word_list, tagging_out_file)
    print("  %s tagging f1-score: %.2f" % (mode, tagging_eval_result['f1']))
    sys.stdout.flush()

    hyp_label_list = []
    ref_label_list = []

    if FLAGS.task['joint'] == 1:

      hyp_label_list = [rev_label_vocab[idx] for idx in hyp_label_idx_list]
      ref_label_list = [rev_label_vocab[idx] for idx in ref_label_idx_list]

      classification_eval_result = classeval(hyp_label_list, ref_label_list, rev_label_vocab.values(),
                                             classification_out_file)
      for label, cm in classification_eval_result.items():
          print("  \t%s accuracy: %.2f  precision: %.2f  recall: %.2f  f1-score: %.2f" % (
          label, 100 * cm["a"], 100 * cm["p"], 100 * cm["r"], 100 * cm["f1"]))
      sys.stdout.flush()

      if extra_rev_vocab:
          extra_hyp_tag_list = []
          extra_ref_tag_list = []
          for current_hyp_tag_idx_list, current_hyp_label_idx in zip(hyp_tag_idx_list, hyp_label_idx_list):
            extra_hyp_tag_list.append([extra_rev_vocab[str((tag_idx, current_hyp_label_idx))]
                                       for tag_idx in current_hyp_tag_idx_list])

          for current_ref_tag_idx_list, current_ref_label_idx in zip(ref_tag_idx_list, ref_label_idx_list):
            extra_ref_tag_list.append([extra_rev_vocab[str((tag_idx, current_ref_label_idx))]
                                       for tag_idx in current_ref_tag_idx_list])

          extra_eval_result = conlleval(extra_hyp_tag_list, extra_ref_tag_list, word_list, tagging_out_file + ".joint")
          print("  %s joint tagging f1-score: %.2f" % (mode, extra_eval_result['f1']))


    if attentions_out_file:
      json_att_dic = {"classification_atts": hyp_class_attentions,
                      "tagging_atts": hyp_label_attentions,
                      "sequences": word_list,
                      "ref_classes": ref_label_list,
                      "hyp_classes": hyp_label_list,
                      "ref_tags": ref_tag_list,
                      "hyp_tags": hyp_tag_list}

      with open(attentions_out_file, "w") as f:
        json.dump(json_att_dic, f)

    if FLAGS.task['joint'] == 0 and extra_rev_vocab:
        hyp_tag_list = []
        ref_tag_list = []

        hyp_label_list = []
        ref_label_list = []

        for gold_sequence in ref_tag_idx_list:
          seq_tags, seq_classes = zip(*map(lambda x: extra_rev_vocab[str(x)], gold_sequence))
          ref_tag_list.append(seq_tags)
          ref_label_list += tagclasses2classes(seq_classes, seq_tags)

        for i, predicted_sequence in enumerate(hyp_tag_idx_list):
          seq_tags, seq_classes = zip(*map(lambda x: extra_rev_vocab[str(x)], predicted_sequence))
          hyp_tag_list.append(seq_tags)
          hyp_label_list += tagclasses2classes(seq_classes, ref_tag_list[i])

        extra_eval_result = conlleval(hyp_tag_list, ref_tag_list, word_list, tagging_out_file + ".simple")
        print("  %s simple tagging f1-score: %.2f" % (mode, extra_eval_result['f1']))
        sys.stdout.flush()

        labels = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
        classification_eval_result = classeval(hyp_label_list, ref_label_list, labels,
                                               classification_out_file + ".simple")

        for label, cm in classification_eval_result.items():
            print("  \tsimple %s accuracy: %.2f  precision: %.2f  recall: %.2f  f1-score: %.2f" % (
            label, 100 * cm["a"], 100 * cm["p"], 100 * cm["r"], 100 * cm["f1"]))
        sys.stdout.flush()

    print("")
    sys.stdout.flush()

    return classification_eval_result, tagging_eval_result, extra_eval_result


if __name__ == "__main__":
  main()
